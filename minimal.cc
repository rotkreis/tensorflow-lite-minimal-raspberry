/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// and compute with a image
//
// Usage: minimal <tflite model> <image>

using namespace tflite;
using namespace std::chrono;
using namespace std;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <image>\n");
    return 1;
  }
  const char* filename = argv[1];
  const string image_path = argv[2];

  // Read image
  auto t0_0 = high_resolution_clock::now();
  auto img = cv::imread(image_path);
  // throw away top 160 pixels, as data pre-processing
  img = img(cv::Rect(0, 160, 640, 320)); 
  cv::resize(img, img, cv::Size(200,66));
  cv::cvtColor(img, img, CV_BGR2RGB);
  img.convertTo(img, CV_32FC3);
  img = (img / 255.0 - 0.5) / 0.5;

  auto t0_1 = high_resolution_clock::now();
  auto duration0 = duration_cast<milliseconds>(t0_1 - t0_0);
  cout << "image processing time : " << duration0.count() << endl;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
//  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
//  int range = 3 * 66 * 200;
//  for (int i = 0; i < range; ++i) {
//	  interpreter->typed_input_tensor<float>(0)[i] = 0.5;
//  }

  int index = 0;
  const int offset = img.rows * img.cols;
  for (int i = 0; i < img.rows; ++i) {
	const auto pixel = img.ptr<cv::Vec3f>(i);
    for(int j = 0; j < img.cols; ++j) {
//			interpreter->typed_input_tensor<float>(0)[index] = pixel[j][0];
//			interpreter->typed_input_tensor<float>(0)[index + 1] = pixel[j][1];
//			interpreter->typed_input_tensor<float>(0)[index + 2] = pixel[j][2];

			interpreter->typed_input_tensor<float>(0)[index] = pixel[j][0];
			interpreter->typed_input_tensor<float>(0)[index + offset] = pixel[j][1];
			interpreter->typed_input_tensor<float>(0)[index + offset * 2] = pixel[j][2];

			index++;
	}
  }
  std::cout << interpreter->typed_input_tensor<float>(0)[0] << " " 
		  << interpreter->typed_input_tensor<float>(0)[1] <<  " "
		  << interpreter->typed_input_tensor<float>(0)[2] << std::endl;

  // Run inference
  auto t1 = high_resolution_clock::now();
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  auto t2 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(t2-t1);
  cout << "Inference timing : " << duration.count() << endl;
  printf("\n\n=== Post-invoke Interpreter State ===\n");
 // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  float* output = interpreter->typed_output_tensor<float>(0);
  std::cout << "result" << *output << std::endl;

  return 0;
}
