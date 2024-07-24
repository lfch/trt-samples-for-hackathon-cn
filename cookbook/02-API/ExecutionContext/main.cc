#include "../common.h"
#include <string>
#include <vector>

using namespace nvinfer1;

const std::string trtFile {"./model.plan"};
static Logger gLogger(ILogger::Severity::kERROR);

int main() {
  ICudaEngine *engine = nullptr;

  IBuilder             *builder = createInferBuilder(gLogger);
  INetworkDefinition   *network = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  IOptimizationProfile *profile = builder->createOptimizationProfile();
  IBuilderConfig       *config  = builder->createBuilderConfig();
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);

  ITensor *inputTensor = network->addInput("inputT0", DataType::kFLOAT, Dims32 {3, {-1, -1, -1}});
  profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {3, {1, 1, 1}});
  profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {3, {3, 4, 5}});
  profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {3, {6, 8, 10}});
  config->addOptimizationProfile(profile);

  IIdentityLayer *identityLayer = network->addIdentity(*inputTensor);
  network->markOutput(*identityLayer->getOutput(0));
  IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
  if (engineString == nullptr || engineString->size() == 0)
  {
      std::cout << "Failed building serialized engine!" << std::endl;
      return 0;
  }
  std::cout << "Succeeded building serialized engine!" << std::endl;

  IRuntime *runtime {createInferRuntime(gLogger)};
  engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
  if (engine == nullptr)
  {
      std::cout << "Failed building engine!" << std::endl;
      return 0;
  }
  std::cout << "Succeeded building engine!" << std::endl;

  std::ofstream engineFile(trtFile, std::ios::binary);
  if (!engineFile)
  {
      std::cout << "Failed opening file to write" << std::endl;
      return 0;
  }
  engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
  if (engineFile.fail())
  {
      std::cout << "Failed saving .plan file!" << std::endl;
      return 0;
  }
  std::cout << "Succeeded saving .plan file!" << std::endl;

  unsigned long int nIO = engine->getNbIOTensors();
  unsigned long int nInput = 0;
  unsigned long int nOutput = 0;
  std::vector<std::string> vTensorName(nIO);

  for (int i = 0; i < nIO; i++) {
    vTensorName[i] = std::string(engine->getIOTensorName(i));
    nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
    nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
  }

  IExecutionContext *context = engine->createExecutionContext();
  // context->setInputShape(vTensorName[0].c_str(), Dims32{3, {3, 4, 5}});

  for (int i = 0; i < nIO; i++) {
    std::cout << std::string(i < nInput ? "Input [" : "Output [");
    std::cout << i << std::string("]-> ");
    std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
    std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
    std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
    std::cout << vTensorName[i] << std::endl;
  }

  return 0;
}