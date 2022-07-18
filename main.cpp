#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
using namespace std;

void compile(bool fp16) {
  constexpr int64_t INPUT_CHANNEL_NUM = 42;
  constexpr int64_t WIDTH = 9;
  torch::jit::Module module = torch::jit::load("shogi_cat_bl10_ch256.ts");
  if (fp16) {
    module.to(torch::kCUDA, torch::kFloat16);
  } else {
    module.to(torch::kCUDA);
  }
  module.eval();

  std::vector<int64_t> in_sizes = {1, INPUT_CHANNEL_NUM, WIDTH, WIDTH};

  auto input =
      (fp16 ? torch_tensorrt::Input(in_sizes, torch::kFloat16) : torch_tensorrt::Input(in_sizes, torch::kFloat32));
  auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
  if (fp16) {
    compile_settings.enabled_precisions = {torch::kFloat16};
  }
  module = torch_tensorrt::ts::compile(module, compile_settings);

  torch::Tensor sample_input = torch::zeros({1, INPUT_CHANNEL_NUM, WIDTH, WIDTH}).to(torch::kCUDA);
  if (fp16) {
    sample_input = sample_input.to(torch::kFloat16);
  }

  auto out = module.forward({sample_input});
  auto tuple = out.toTuple();
  torch::Tensor policy = tuple->elements()[0].toTensor();
  torch::Tensor value = tuple->elements()[1].toTensor();
  std::cout << policy.sizes() << " " << value.sizes() << std::endl;
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