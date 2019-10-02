#include <stdio.h>
#include <stdlib.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\video.hpp>

using namespace std;
using namespace cv;

struct TomatoeEvent
{
	TomatoeEvent() {

	}

	TomatoeEvent(int frame, Point2f coord, bool isVisible) {
		TomatoeEvent::coord = coord;
		TomatoeEvent::frame = frame;
		TomatoeEvent::isVisible = isVisible;
	}

	Point2f coord;
	int frame;
	int id;
	bool isVisible;
};

VideoCapture video_in;
std::vector<TomatoeEvent> tomatoeEvents;
std::vector<cv::Point2f> p0, p1;
std::vector<Point2f> tomatoes;
int w_size = 15;
int currentFrame = -1;

int x_max_lbl = 1700;	// Max X position for tomato labels
Point2f footer_lbl_p = Point2f(20, 1050);
string footer_lbl;

static void help()
{
	printf("\nThis is a demo of Lukas-Kanade optical flow:\n");
	printf("Using OpenCV version %s", CV_VERSION);
	printf("\nHot keys: \n");
	printf("\tESC - quit the program\n");
	printf("\tf - input a filename(path) and load stored feature points\n");
	printf("\tp - select/unselect feature points and store it in a file:\n");
	printf("\t\tTo add a feature point double (left) click it\n");
	printf("\t\tTo remove an existing feature point double (right) click it\n");
	printf("\t\td - switch the \"draw circles\" mode on/off\n");
	printf("\n\n");
}

// Mouse event
static void onMouse(int event, int x, int y, int, void*)
{
	static int T_id = 0;
	if (event == EVENT_LBUTTONDBLCLK)
	{
		//p0[T_id] = cv::Point2f(x, y);
		//T_id++;
		//printf("Frame %d Tomato %d in coordenates (%d, %d).\n", currentFrame,T_id, x, y);
		//if (T_id >= numTomatos) T_id = 0;

		tomatoes.push_back(Point2f(x, y));
		tomatoeEvents.push_back(TomatoeEvent(currentFrame, Point2f(x, y), true));
		printf("Frame %d Tomato added in coordenates (%d, %d).\n", currentFrame, x, y);
	}
	else if (event == EVENT_RBUTTONDBLCLK)
	{
		/*for (int i = 0; i < numTomatos; i++)
		{
		if ((x >= (p0[i].x - w_size)) && (x <= (p0[i].x + w_size)) && (y >= (p0[i].y - w_size)) && (y <= (p0[i].y + w_size)))
		{
		p0[i].x = 0;
		p0[i].y = 0;
		p1[i].x = 0;
		p1[i].y = 0;
		printf("Deleting tomato %d in coordenates (%d, %d).\n", i, x, y);
		}
		}*/

		//for (int i = 0; i < tomatoes.size(); i++){
		//if ((x >= (tomatoes[i].x - w_size)) && (x <= (tomatoes[i].x + w_size)) && (y >= (tomatoes[i].y - w_size)) && (y <= (tomatoes[i].y + w_size))) {
		tomatoeEvents.push_back(TomatoeEvent(currentFrame, Point2f(x, y), false));
		printf("Frame %d Tomato removed in coordenates (%d, %d).\n", currentFrame, x, y);
		//}
		//}
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
	int ex = static_cast<int>(video_in.get(CAP_PROP_FOURCC));    // Get Codec Type

	VideoWriter outputVideo;  // Open the output
	printf("FPS: %f", video_in.get(CAP_PROP_FPS));
	outputVideo.open("test_output.mp4", ex, video_in.get(CAP_PROP_FPS), S, true);

	if (!outputVideo.isOpened()) {
		printf("Could not open the output video for write.\n");
		return -1;
	}

	printf("Saving video file...\n");
	int n_frames = images.size();
	if (images.size() > 900) {
		n_frames = 900;
	}
		
	for (int i = 0; i < n_frames; i++)
		outputVideo << images[i];

	outputVideo.release();
	return 0;
}

void PickTomatoes(VideoCapture &video_in, std::vector<TomatoeEvent> &events, int circleSize) {
	Mat frame, frame_gray, old_frame, old_gray;	
	bool drawCircles = false;
	currentFrame = 0;

	video_in.set(CAP_PROP_POS_FRAMES, 0);
	video_in.read(old_frame);
	cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
	//currentFrame++;

	while (true) {
		video_in.read(frame);
		if (frame.empty()) {
			break;
		}
	
		// Compute the optical flow and shows the circles
		if (drawCircles && events.size() > 0)
		{
			for (int currentEvent = 0; currentEvent < events.size(); currentEvent++) {
				if (currentFrame == events[currentEvent].frame) {
					if (events[currentEvent].isVisible) {
						p0.push_back(events[currentEvent].coord);
						printf("Tomato Added in (%d, %d)\n", (int)events[currentEvent].coord.x, (int)events[currentEvent].coord.y);
					}
					else {
						float minDistance = INFINITY;
						int nearPoint = -1;
						for (int i = 0; i < p1.size(); i++) {
							float distance = cv::norm(p1[i] - events[currentEvent].coord);
							if (distance < minDistance) {
								nearPoint = i;
								minDistance = distance;
							}
						}
						if (nearPoint >= 0) {
							p0.erase(p0.begin() + nearPoint);
							printf("Tomato deleted from (%d, %d)\n", (int)events[currentEvent].coord.x, (int)events[currentEvent].coord.y);
						}
					}
				}
			}
			p1.resize(p0.size());
			cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

			vector<uchar> status;
			vector<float> err;
			TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

			if (p0.size() > 0) {
				string label;
				Point2f lbl_p;

				// calculate optical flow
				calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(w_size, w_size), 2, criteria);

				// Print circles and labels
				for (int i = 0; i < p1.size(); i++) {
					if (status[i]) {
						label = "Tomato " + to_string(i + 1) + " pos(" + to_string((int)p1[i].x) + ", " + to_string((int)p1[i].y) + ")";
						circle(frame, p1[i], circleSize, cv::Scalar(255, 0, 0, 255), 3);
						lbl_p = p1[i];
						if ((int)lbl_p.x > 1700) lbl_p.x = 1700;
						cv::putText(frame, label, lbl_p, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
					}
				}
			}

			footer_lbl = "Frame: " + to_string(currentFrame++) + " (" + to_string(p1.size()) + " tomatoes)";
			cv::putText(frame, footer_lbl, footer_lbl_p, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.8);
			cv::imshow("Video", frame);	// Show modified frame
			images.push_back(frame);	// Store new frame in output video

			if (waitKey(10) == 27) {
				printf("Processing interruped\n");
				break;
			}

			old_gray = frame_gray.clone();

			for (int i = 0; i < p0.size(); i++) {
				p0[i] = p1[i];
			}
			printf("%s\n", footer_lbl.c_str());
		}
		else
		{
			cv::imshow("Video", frame);        // Show new frame	
			printf("Frame: %d\n", currentFrame);
			images.push_back(frame);
			currentFrame++;
		}

	/*	if (currentFrame == 0) {
			waitKey();
		}*/		
		int key = waitKey(0);
		if (key == 27) {
			break;
		}
		else if (key == 100) { //d
			drawCircles = !drawCircles;
		}			 
	}
}

void ComputeOpticalFlow(VideoCapture &video, std::vector<TomatoeEvent> events, int circleSize) {
	Mat frame, frame_gray, old_frame, old_gray;
	std::vector<Point2f> p0, p1;
	currentFrame = 0;

	video_in.set(CAP_PROP_POS_FRAMES, 0);
	video_in.read(old_frame);
	cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
	currentFrame++;

	while (true) {
		//// KK
		//if (currentFrame == 774) {
		//	printf("yei");
		//}

		for (int currentEvent = 0; currentEvent < events.size(); currentEvent++) {
			if (currentFrame == events[currentEvent].frame) {
				if (events[currentEvent].isVisible) {
					p0.push_back(events[currentEvent].coord);
					printf("Tomato Added in (%d, %d)\n", (int)events[currentEvent].coord.x, (int)events[currentEvent].coord.y);
				}
				else {
					float minDistance = INFINITY;
					int nearPoint = -1;
					for (int i = 0; i < p1.size(); i++) {
						float distance = cv::norm(p1[i] - events[currentEvent].coord);
						if (distance < minDistance) {
							nearPoint = i;
							minDistance = distance;
						}
					}
					if (nearPoint >= 0 && nearPoint < p0.size()) {
						p0.erase(p0.begin() + nearPoint);
						printf("Tomato deleted from (%d, %d)\n", (int)events[currentEvent].coord.x, (int)events[currentEvent].coord.y);
					}
				}
			}
		}
		p1.resize(p0.size());

		video_in.read(frame);

		if (frame.empty()) {
			break;
		}
		cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

		if (p0.size() > 0) {
			string label;
			Point2f lbl_p;

			// calculate optical flow
			calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(w_size, w_size), 2, criteria);

			// Print circles and labels
			for (int i = 0; i < p1.size(); i++) {
				if (status[i]) {
					label = "Tomato " + to_string(i + 1) + " pos(" + to_string((int)p1[i].x) + ", " + to_string((int)p1[i].y) + ")";
					circle(frame, p1[i], circleSize, cv::Scalar(255, 0, 0, 255), 3);
					lbl_p = p1[i];
					if ((int)lbl_p.x > 1700) lbl_p.x = 1700;
					cv::putText(frame, label, lbl_p, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.0);
				}				
			}
		}

		footer_lbl = "Frame " + to_string(currentFrame++) + " (" + to_string(p1.size()) + " tomatoes)";
		cv::putText(frame, footer_lbl, footer_lbl_p, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 1.8);
		cv::imshow("Video", frame);	// Show new frame
		images.push_back(frame);	// Store new frame in output video

		if (waitKey(10) == 27) {
			printf("Processing interruped\n");
			break;
		}

		old_gray = frame_gray.clone();

		for (int i = 0; i < p0.size(); i++) {
			p0[i] = p1[i];
		}
		printf("%s\n", footer_lbl.c_str());
	}
}

void saveTomatoEvents(std::vector<TomatoeEvent> events, const char *filename) {
	FILE *file;
	fopen_s(&file, filename, "w");
	// TODO : Chek if the file opens correctly

	for (int i = 0; i < events.size(); i++) {
		fprintf(file, "Frame: %d\t Coord: %d %d\t%d\n", events[i].frame, (int)(events[i].coord.x), (int)(events[i].coord.y), (int)(events[i].isVisible));
	}
	fclose(file);
}

void readTomatoEvents(std::vector<TomatoeEvent> &events, const char *filename) {
	FILE *file;
	fopen_s(&file, filename, "r");
	// TODO : Chek if the file opens correctly
	char buffer[200];

	while (fgets(buffer, sizeof(buffer), file)) {
		events.push_back(TomatoeEvent());
		sscanf_s(buffer, "Frame: %d\t Coord: %f %f\t%d\n", &events[events.size() - 1].frame, &events[events.size() - 1].coord.x, &events[events.size() - 1].coord.y, &events[events.size() - 1].isVisible);
	}
	fclose(file);
}

int main(int argc, char** argv)
{
	help();

	// TODO : Also ask for enter path
	char path[] = "Data/";

	char fileName[] = "test.mp4";
	//char fileName[100];
	//printf("Video filename in %s : ",path);
	//scanf_s("%s", fileName, sizeof(fileName));

	char fullPath[200];
	sprintf_s(fullPath, sizeof(fullPath), "%s%s", path, fileName);

	// Load video
	video_in.open(fullPath);

	// Check if video opened successfully
	if (!video_in.isOpened()) {
		printf("Error opening video stream or file.\n");
		return -1;
	}

	namedWindow("Video", 0);
	setMouseCallback("Video", onMouse, 0);

	// Pick manually the tomatoes in the video
	// TODO : Ask for pick or read file
	int key = waitKey();
	if (key == 'p') {
		PickTomatoes(video_in, tomatoeEvents, 40);

		// Serialize and save
		char selectionFileName[] = "manualSelection.txt";
		//char selectionFileName[100];
		//printf("Enter filename for save selection : ");
		//scanf_s("%s", fileName, sizeof(selectionFileName));

		sprintf_s(fullPath, sizeof(fullPath), "%s%s", path, selectionFileName);
		saveTomatoEvents(tomatoeEvents, fullPath);
	}
	else if (key == 'f') {
		// TODO : Check
		char selectionFileName[100];
		printf("Enter filename for read selection: ");
		scanf_s("%s", selectionFileName, sizeof(selectionFileName));

		sprintf_s(fullPath, sizeof(fullPath), "%s%s", path, selectionFileName);
		readTomatoEvents(tomatoeEvents, fullPath);
	}
	else {
		exit(0);
	}

	printf("Press 'c' for compute or any key for continue.\n");
	if (waitKey() == 'c') {
		try	{
			ComputeOpticalFlow(video_in, tomatoeEvents, 40);
		}
		catch (std::exception& e) {
			printf("Standard exception: %s\n", e.what());
		}
	}

	// Ask confirmation fot save or not
	printf("Press 's' for save the new video file or any key for exit.\n");
	if (waitKey() != 's') {
		exit(0);
	}

	if (SaveVideo() >= 0) {
		printf("Output video saved successfully.\n");
	}
	else {
		printf("Error saving the new video stream to file.\n");
	}
	waitKey();
}
