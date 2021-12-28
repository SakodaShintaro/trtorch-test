#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
using namespace std;

void compile(bool fp16) {
  torch::jit::Module module = torch::jit::load("model.ts");
  if (fp16) {
    module.to(torch::kCUDA, torch::kHalf);
  } else {
    module.to(torch::kCUDA);
  }
  module.eval();

  constexpr int64_t INPUT_CHANNEL_NUM = 256;
  constexpr int64_t WIDTH = 32;
  std::vector<int64_t> in_sizes = {1, WIDTH, INPUT_CHANNEL_NUM};
  torch_tensorrt::Input range(in_sizes);
  torch_tensorrt::ts::CompileSpec info({range});
  if (fp16) {
    info.enabled_precisions.insert(torch::kHalf);
  } else {
    info.enabled_precisions.insert(torch::kF32);
  }
  module = torch_tensorrt::ts::compile(module, info);
}

int main() {
  // fp32, this thread -> OK
  compile(false);
  cout << "fp32, this thread -> finish" << endl;

  // fp32, another thread -> OK
  std::thread thread0([]() { compile(false); });
  thread0.join();
  cout << "fp32, another thread -> finish" << endl;

  // fp16, this thread -> OK
  compile(true);
  cout << "fp16, this thread -> finish" << endl;

  // fp16, another thread -> NG
  std::thread thread1([]() { compile(true); });
  thread1.join();
  cout << "fp16, another thread -> finish" << endl;
}