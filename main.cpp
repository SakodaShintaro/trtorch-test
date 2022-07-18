#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
using namespace std;

void compile() {
  constexpr int64_t INPUT_CHANNEL_NUM = 42;
  constexpr int64_t WIDTH = 9;
  torch::jit::Module module = torch::jit::load("shogi_cat_bl10_ch256.ts");
  module.to(torch::kCUDA, torch::kFloat16);
  module.eval();

  const int64_t kOptBatchSize = 8;
  std::vector<int64_t> in_min = {1, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  std::vector<int64_t> in_opt = {kOptBatchSize, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  std::vector<int64_t> in_max = {kOptBatchSize * 2, INPUT_CHANNEL_NUM, WIDTH, WIDTH};

  auto input = torch_tensorrt::Input(in_min, in_opt, in_max, torch::kFloat16);
  auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
  compile_settings.enabled_precisions = {torch::kFloat16};
  module = torch_tensorrt::ts::compile(module, compile_settings);

  for (int64_t batch_size : {(int64_t)1, kOptBatchSize, kOptBatchSize * 2}) {
    torch::Tensor sample_input =
        torch::zeros({batch_size, INPUT_CHANNEL_NUM, WIDTH, WIDTH}).to(torch::kCUDA, torch::kFloat16);
    auto out = module.forward({sample_input});
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor();
    torch::Tensor value = tuple->elements()[1].toTensor();
    std::cout << policy.sizes() << " " << value.sizes() << std::endl;
  }
}

int main() {
  compile();
  cout << "fp16, this thread -> finish" << endl;

  std::thread thread1([]() { compile(); });
  thread1.join();
  cout << "fp16, another thread -> finish" << endl;
}
