#include <torch/script.h>
#include <torch_tensorrt/torch_tensorrt.h>
using namespace std;

constexpr int64_t INPUT_CHANNEL_NUM = 42;
constexpr int64_t WIDTH = 9;
constexpr c10::ScalarType kScalarType = torch::kFloat16;
constexpr int64_t kOptBatchSize = 8;

torch::jit::Module compile(int64_t gpu_id) {
  torch::jit::Module module = torch::jit::load("shogi_cat_bl10_ch256.ts");
  const torch::Device device(torch::kCUDA, gpu_id);

  module.to(device, kScalarType);
  module.eval();

  std::vector<int64_t> in_min = {1, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  std::vector<int64_t> in_opt = {kOptBatchSize, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  std::vector<int64_t> in_max = {kOptBatchSize * 2, INPUT_CHANNEL_NUM, WIDTH, WIDTH};

  auto input = torch_tensorrt::Input(in_min, in_opt, in_max, kScalarType);
  auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
  compile_settings.enabled_precisions = {kScalarType};
  module = torch_tensorrt::ts::compile(module, compile_settings);
  return module;
}

void infer(torch::jit::Module& module, int64_t gpu_id) {
  const torch::Device device(torch::kCUDA, gpu_id);
  for (int64_t batch_size : {(int64_t)1, kOptBatchSize, kOptBatchSize * 2}) {
    torch::Tensor sample_input =
        torch::zeros({batch_size, INPUT_CHANNEL_NUM, WIDTH, WIDTH}).to(device, kScalarType);
    auto out = module.forward({sample_input});
    auto tuple = out.toTuple();
    torch::Tensor policy = tuple->elements()[0].toTensor().cpu();
    torch::Tensor value = tuple->elements()[1].toTensor().cpu();
    std::cout << policy.sizes() << " " << value.sizes() << std::endl;
  }
}

int main() {
  torch::jit::Module module0 = compile(0);
  torch::jit::Module module1 = compile(1);

  std::thread thread0([&]() { infer(module0, 0); });
  std::thread thread1([&]() { infer(module1, 1); });
  thread0.join();
  thread1.join();
}
