#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

struct Net_config {
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
	string datatype;
};

typedef struct BoxInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class FreeYOLO
{

public:
    FreeYOLO(Net_config config);
    void detect(Mat& frame);

private:

    ONNXTensorElementDataType inputType;
    ONNXTensorElementDataType outputType;

    std::vector<std::string> inputNamesString;
    std::vector<std::string> outputNamesString;

    vector<const char*> input_names;
	vector<const char*> output_names;

    void getInputDetails(Ort::AllocatorWithDefaultOptions allocator);
    void getOutputDetails(Ort::AllocatorWithDefaultOptions allocator);

	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;
	const int num_stride = 3;
	int strides[3] = { 8,16,32 };

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "FreeYOLO");
	Ort::Session session{nullptr};;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

FreeYOLO::FreeYOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	string model_path = config.modelpath;

    session = Ort::Session(env, model_path.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;
    this->getInputDetails(allocator);
    this->getOutputDetails(allocator);

    if (config.datatype == "coco")
	{
		string classesFile = "coco.names";
		ifstream ifs(classesFile.c_str());
		string line;
		while (getline(ifs, line)) this->class_names.push_back(line);
	}
	else if (config.datatype == "face")
	{
		this->class_names.push_back("face");
	}
	else
	{
		this->class_names.push_back("person");

	}
	this->num_class = class_names.size();
}

void FreeYOLO::detect(Mat& frame)
{
	const float ratio = std::min(float(this->inpHeight) / float(frame.rows), float(this->inpWidth) / float(frame.cols));
	const int neww = int(frame.cols * ratio);
	const int newh = int(frame.rows * ratio);

	Mat dstimg;
	resize(frame, dstimg, Size(neww, newh));
	copyMakeBorder(dstimg, dstimg, 0, this->inpHeight - newh, 0, this->inpWidth - neww, BORDER_CONSTANT, 114);

	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	vector<BoxInfo> generate_boxes;

	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	int n = 0, i = 0, j = 0, k = 0; ///cx, cy, w, h, box_score, class_score
	for (n = 0; n < this->num_stride; n++)
	{
		int num_grid_x = (int)ceil((this->inpWidth / strides[n]));
		int num_grid_y = (int)ceil((this->inpHeight / strides[n]));
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				const float box_score = pdata[4];
				int max_ind = 0;
				float max_class_socre = 0;
				for (k = 0; k < num_class; k++)
				{
					if (pdata[k + 5] > max_class_socre)
					{
						max_class_socre = pdata[k + 5];
						max_ind = k;
					}
				}
				max_class_socre *= box_score;
				max_class_socre = sqrt(max_class_socre);

				if (max_class_socre > this->confThreshold)
				{
					float cx = (0.5f + j + pdata[0]) * strides[n];  ///cx
					float cy = (0.5f + i + pdata[1]) * strides[n];   ///cy
					float w = expf(pdata[2]) * strides[n];   ///w
					float h = expf(pdata[3]) * strides[n];  ///h

					float xmin = (cx - 0.5 * w) / ratio;
					float ymin = (cy - 0.5 * h) / ratio;
					float xmax = (cx + 0.5 * w) / ratio;
					float ymax = (cy + 0.5 * h) / ratio;

					generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
				}
				pdata += nout;
			}
		}		

	}
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);

	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

void FreeYOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void FreeYOLO::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}

void FreeYOLO::getInputDetails(Ort::AllocatorWithDefaultOptions allocator)
{
    inputType = this->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    cout << "---------------- Input info --------------";
    for (int layer=0; layer < this->session.GetInputCount(); layer+=1)
    {
        #if ORT_API_VERSION < 13
            inputNames.push_back(this->session.GetInputName(layer, allocator));
        #else
            Ort::AllocatedStringPtr input_name_Ptr = this->session.GetInputNameAllocated(layer, allocator);
            inputNamesString.push_back(input_name_Ptr.get());
            input_names.push_back(inputNamesString[layer].c_str());
        #endif
        cout << "Name [" << layer << "]: " << input_names[layer];

        std::vector<int64_t> inputTensorShape = this->session.GetInputTypeInfo(layer).GetTensorTypeAndShapeInfo().GetShape();

        input_node_dims.push_back(inputTensorShape);
        cout << "Shape [" << layer << "]: (" << "";
        for (const int64_t& shape : inputTensorShape)
            cout << shape << ", ";
        cout << ")";
    }
    this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void FreeYOLO::getOutputDetails(Ort::AllocatorWithDefaultOptions allocator)
{
    outputType = this->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    cout << "--------------- Output info --------------";
    for (int layer=0; layer < this->session.GetOutputCount(); layer+=1)
    {
        #if ORT_API_VERSION < 13
            outputNames.push_back(this->session.GetOutputName(layer, allocator));
        #else
            Ort::AllocatedStringPtr output_name_Ptr = this->session.GetOutputNameAllocated(layer, allocator);
            outputNamesString.push_back(output_name_Ptr.get());
            output_names.push_back(outputNamesString[layer].c_str());
        #endif
        cout << "Name [" << layer << "]: " << output_names[layer];
        
        auto outputTensorShape = this->session.GetOutputTypeInfo(layer).GetTensorTypeAndShapeInfo().GetShape();
        output_node_dims.push_back(outputTensorShape);
        cout << "Shape [" << layer << "]: (" << "";
        for (const int64_t& shape : outputTensorShape)
            cout << shape << ", ";
        cout << ")";
    }
}

int main()
{
	Net_config cfg = { 0.6, 0.5, "weights/coco/yolo_free_nano_192x320.onnx", "coco" };
	FreeYOLO net(cfg);
	string imgpath = "images/coco/dog.jpg";
	Mat srcimg = imread(imgpath);
	net.detect(srcimg);
	imwrite("res_cpp.jpg", srcimg);
	// static const string kWinName = "Deep learning object detection in ONNXRuntime";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	// waitKey(0);
	// destroyAllWindows();
}
