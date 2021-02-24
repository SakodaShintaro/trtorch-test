#include <torch/torch.h>
#include <torch/script.h>
#include <trtorch/trtorch.h>
#include <trtorch/ptq.h>
using namespace std;

constexpr int64_t INPUT_CHANNEL_NUM = 256;
constexpr int64_t WIDTH = 32;

enum Precision { INT8, FP16, FP32 };

class CalibrationDataset
    : public torch::data::datasets::Dataset<CalibrationDataset> {
public:
  explicit CalibrationDataset(int64_t data_num) {
    for (int64_t i = 0; i < data_num; i++) {
      // input
      data_.push_back(torch::randn({1, INPUT_CHANNEL_NUM, WIDTH, WIDTH}));

      // target(dummy data)
      targets_.push_back(torch::tensor({0}));
    }
  }

  torch::data::Example<> get(size_t index) override {
    return {data_[index].clone().to(torch::kCUDA),
            targets_[index].clone().to(torch::kCUDA)};
  }

  c10::optional<size_t> size() const override { return data_.size(); }

private:
  std::vector<torch::Tensor> data_, targets_;
};

void compile(Precision p) {
  torch::jit::Module module = torch::jit::load("model.ts");
  if (p == FP16) {
    module.to(torch::kCUDA, torch::kHalf);
  } else {
    module.to(torch::kCUDA);
  }
  module.eval();

  const int64_t batch_size = 1;

  std::vector<int64_t> in_sizes = {batch_size, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  trtorch::CompileSpec::InputRange range(in_sizes);
  trtorch::CompileSpec info({range});
  if (p == INT8) {
    info.op_precision = torch::kInt8;

    using namespace torch::data;
    auto dataset =
        CalibrationDataset(batch_size * 2).map(transforms::Stack<>());
    auto dataloader =
        make_data_loader(move(dataset),
                         DataLoaderOptions().batch_size(batch_size).workers(1));

    const std::string name = "calibration_cache.txt";
    auto calibrator =
        trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(
            move(dataloader), name, false);

    info.ptq_calibrator = calibrator;
    info.workspace_size = (1ull << 28);
    info.max_batch_size = batch_size;
    module = trtorch::CompileGraph(module, info);
  } else if (p == FP16) {
    info.op_precision = torch::kHalf;
    module = trtorch::CompileGraph(module, info);
  } else {
    module = trtorch::CompileGraph(module, info);
  }
}

int main() {
  // fp32, this thread -> OK
  compile(FP32);
  cout << "fp32, this thread -> finish" << endl;

  // fp32, another thread -> OK
  std::thread thread0([]() { compile(FP32); });
  thread0.join();
  cout << "fp32, another thread -> finish" << endl;

  // fp16, this thread -> OK
  compile(FP16);
  cout << "fp16, this thread -> finish" << endl;

  // fp16, another thread -> NG
  std::thread thread1([]() { compile(FP16); });
  thread1.join();
  cout << "fp16, another thread -> finish" << endl;

  // int8, this thread
  compile(INT8);
  cout << "int8, this thread -> finish" << endl;

  // int8, another thread
  std::thread thread2([]() { compile(INT8); });
  thread2.join();
  cout << "int8, another thread -> finish" << endl;
}