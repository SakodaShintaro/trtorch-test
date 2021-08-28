#include <torch/script.h>
#include <trtorch/trtorch.h>
using namespace std;

int main() {
  constexpr int64_t INPUT_CHANNEL_NUM = 256;
  constexpr int64_t IMAGE_SIZE = 32;
  torch::jit::Module module = torch::jit::load("model.ts");
  module.to(torch::kCUDA);
  module.eval();

  std::vector<int64_t> in_sizes = {1, INPUT_CHANNEL_NUM, IMAGE_SIZE, IMAGE_SIZE};
  trtorch::CompileSpec::Input range(in_sizes);
  trtorch::CompileSpec info({range});
  module = trtorch::CompileGraph(module, info);
  cout << "finish" << endl;
}