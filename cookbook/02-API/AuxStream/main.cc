#include "../common.h"

#include <string>
#include <vector>
#include <memory>

using namespace nvinfer1;

const int nGEMM = 10;
const int nMKN = 128;
const std::string trtFile {"./model.plan"};

static Logger gLogger(ILogger::Severity::kVERBOSE);

int main() {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition * network = builder->createNetworkV2(1 << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
    config->setMaxAuxStreams(2);

    std::vector<ITensor*>  inputList;
    inputList.reserve(nGEMM);

    for (auto i = 0; i < nGEMM + 1; i++) {
        const auto &inputName = std::string("inputT") + std::to_string(i);
        auto *input = network->addInput(inputName.c_str(), DataType::kFLOAT, Dims32{4, {-1, 4, nMKN, nMKN}});
        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 4, nMKN, nMKN}});
        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 4, nMKN, nMKN}});
        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 4, nMKN, nMKN}});
        inputList.emplace_back(input);
    }

    config->addOptimizationProfile(profile);

    auto *tempTensor0 = inputList[0];
    auto *tempTensor1 = inputList[1];

    for (auto i = 1; i < nGEMM + 1; i++) {
        auto *tempLayer0 = network->addMatrixMultiply(*tempTensor0, MatrixOperation::kNONE, *(inputList[i]), MatrixOperation::kNONE);
        tempTensor0 = tempLayer0->getOutput(0);
        auto *tempLayer1 = network->addMatrixMultiply(*tempTensor1, MatrixOperation::kNONE, *(inputList[nGEMM + 1 - i]), MatrixOperation::kNONE);
        tempTensor1 = tempLayer1->getOutput(0);
    }

    network->markOutput(*tempTensor0);
    network->markOutput(*tempTensor1);

    IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
    if (engineString == nullptr || engineString->size() == 0) {
        std::cout << "failed building serialized engine" << std::endl;
        return -1;
    }

    IRuntime *runtime = createInferRuntime(gLogger);
    
    auto *engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
    if (engine == nullptr) {
        std::cout << "failed desrializing cuda engine" << std::endl;
        return -1;
    }

    std::ofstream engineFile(trtFile, std::ios::binary);
    if (!engineFile) {
        std::cout << "failed open file to write" << std::endl;
        return -1;
    }
    engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
    if (engineFile.fail()) {
        std::cout << "failed saving plan file" << std::endl;
        return -1;
    }


    return 0;
}