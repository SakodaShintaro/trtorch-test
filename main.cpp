#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
using namespace std;

void compile(bool fp16) {
  auto type = (fp16 ? torch::kF16 : torch::kF32);
  torch::jit::Module module = torch::jit::load("model.ts");
  module.to(torch::kCUDA, type);
  module.eval();

  constexpr int64_t INPUT_CHANNEL_NUM = 256;
  constexpr int64_t WIDTH = 32;
  std::vector<int64_t> in_sizes = {1, WIDTH, INPUT_CHANNEL_NUM};
  torch_tensorrt::Input range(in_sizes, type);
  torch_tensorrt::ts::CompileSpec info({range});
  info.enabled_precisions.insert(type);
  module = torch_tensorrt::ts::compile(module, info);
}

int main() {
  // fp32, this thread -> NG
  compile(false);
  cout << "fp32, this thread -> finish" << endl;

  // fp16, this thread -> NG
  compile(true);
  cout << "fp16, this thread -> finish" << endl;
}