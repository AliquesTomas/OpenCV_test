#include <stdio.h>
#include <stdlib.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>

using namespace std;
using namespace cv;

VideoCapture video_in("C:/Users/Toni/Desktop/Agrobotics/IMG_3021.MOV");

int numTomatos = 10;
std::vector<cv::Point2f> p0(numTomatos), p1(numTomatos);
int w_size = 15;


// Mouse event 
static void onMouse(int event, int x, int y, int, void*)
{
	static int T_id = 0;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		p0[T_id] = cv::Point2f(x, y);
		T_id++;
		printf("Tomato %d in coordenates (%d, %d).\n", T_id, x, y);
		if (T_id >= numTomatos) T_id = 0;
	}
	else if (event == EVENT_RBUTTONDBLCLK)
	{
		for (int i = 0; i < numTomatos; i++)
		{
			if ((x >= (p0[i].x - w_size)) && (x <= (p0[i].x + w_size)) &&
				(y >= (p0[i].y - w_size)) && (y <= (p0[i].y + w_size)))
			{
				p0[i].x = 0;
				p0[i].y = 0;
				p1[i].x = 0;
				p1[i].y = 0;
				printf("Deleting tomato %d in coordenates (%d, %d).\n", i, x, y);
			}
		}
	}
	else
		return;	
}

// Output video
vector<Mat> images; // output video frames

int SaveVideo()
{
	if (images.empty())
		return -1;

	Size S = Size((int)video_in.get(CAP_PROP_FRAME_WIDTH),    
		(int)video_in.get(CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(video_in.get(CAP_PROP_FOURCC));	// Get Codec Type

	VideoWriter outputVideo;  // Open the output
	outputVideo.open("test_output.mp4", ex, video_in.get(CAP_PROP_FPS), S, true);

	if (!outputVideo.isOpened()) {
		printf("Could not open the output video for write.\n");
		return -1;
	}

	for (int i = 0; i<images.size(); i++) 
		outputVideo << images[i];

	outputVideo.release();
	return 0;
}

int main(int argc, char** argv)
{
	printf("Starting...\n");

	
	// Check if video opened successfully
	if (!video_in.isOpened()) {
		printf("Error opening video stream or file.\n");
		return -1;
	}

	int tomatoSize = 50;

	Mat old_gray, old_frame;

	video_in.read(old_frame);
	cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);

	namedWindow("Test", 0);
	setMouseCallback("Test", onMouse, 0);

	int frameNum = 0;
	// Camera test
	while (true) {
		string label;
		Mat frame, frame_gray;

		video_in.read(frame);

		if (frame.empty())
			break;
		cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

		// calculate optical flow
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

		//ignorar casos no inicializados en vector
		for (int i = 0; i < numTomatos; i++)
		{
			if ((p0[i].x > 0) && (p0[i].y > 0))	// Ignore points not initialized
			{
				calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(w_size, w_size), 2, criteria);
				if (status[i])
				{
					label = "Tomato " + to_string(i+1) + " pos(" + to_string((int)p1[i].x) + ", " + to_string((int)p1[i].y) + ")";
					circle(frame, p1[i], tomatoSize, cv::Scalar(255, 0, 0, 255), 5);
					putText(frame, label, p1[i], FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.9);
				}
				p0[i] = p1[i];
			}
		}
		
		imshow("Test", frame);		// Show new frame
		images.push_back(frame);	// Store new frame in output video
		if (waitKey(10) == 27)
			break;

		// Now update the previous frame and previous points
		old_gray = frame_gray.clone();
		printf("Frame %d\n", frameNum++);
	}

	if (waitKey(0) == 's') // s - Save
	{
		if (SaveVideo() >= 0)
			printf("Output video saved successfully.\n");
		else
			printf("Error saving the new video stream to file.\n");
	}
	printf("Press any key to close.\n");
	waitKey(0);
}