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
    std::random_device rd;
    std::mt19937 gen(rd);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

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

    auto nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    int nInput = 0;
    int nOutput = 0;

    for (int i = 0; i < nIO; i++) {
        vTensorName[i] = engine->getIOTensorName(i);
        nInput += (engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    IExecutionContext *context = engine->createExecutionContext();
    if (context == nullptr) {
        std::cout << "create exeuction context failed" << std::endl;
        return -1;
    }

    for (int i = 0; i < nInput; i++) {
        context->setInputShape(vTensorName[i].c_str(), Dims32 {4, {4, 4, nMKN, nMKN}});
    }

    for (int i = 0; i < nIO; ++i) {
        std::cout << std::string(i < nInput ? "Input [" : "Output [");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    int nAuxStream = engine->getNbAuxStreams();
    std::vector<cudaStream_t> vCudaStream(nAuxStream);
    for (int i = 0; i < nAuxStream; i++) {
        auto err = cudaStreamCreate(&vCudaStream[i]);
        if (err != cudaSuccess) {
            std::cout << "cuda stream create err = " << err << std::endl;
            return -1;
        }    
    }
    context->setAuxStreams(vCudaStream.data(), vCudaStream.size());

    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; i++) {
        Dims32 dim = context->getTensorShape(vTensorName[i].c_str());
        int size = 1;
        for (int j = 0; j < dim.nbDims; j++) {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    for (auto size : vTensorSize) {
        std::cout << "tensor size = " << size << std::endl;
    }

    std::vector<void *> vBufferH {nIO, nullptr};
    std::vector<void *> vBufferD {nIO, nullptr};
    for (int i = 0; i <  nIO; i++) {
        vBufferH[i] = static_cast<void *>(new char[vTensorSize[i]]]);
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    // random generate input between [-1, 1]
    for (int i = 0; i < nInput; i++) {
        float *pData = static_cast<float*>(vBufferH[i])
        size_t typeSize = dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
        for (int j = 0; j < vTensorSize[i] / typeSize; j++) {
            pData[j] = static_cast<float>(dis(gen))
        }

    }

    for (int i = 0; i < nInput; i++) {
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < nIO; i++) {
        context->setTensorAddress(vTensorName[i], vBufferD[i]);
    }

    context->enqueueV(0);

    for (int i = nInput; i < nIO; i++) {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorName[i], cudaMemcpyDeviceToHost));
    }

    for (int i = nInput; i < nIO; i++) {
        printArrayInfo(static_cast<float*>(vBufferH[i]), context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, true);
    }

    return 0;
}