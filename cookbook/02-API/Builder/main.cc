#include "../common.h"

using namespace nvinfer1;

static Logger gLogger(ILogger::Severity::kVERBOSE);

int main() {
    IBuilder *builder = createInferBuilder(gLogger);
    INetworkDefinition *network = builder->createNetworkV2(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    IBuilderConfig *config = builder->createBuilderConfig();

    ITensor *input = network->addInput("inputT0", DataType::kFLOAT, Dims32{3, {3, 4, 5}});
    IIdentityLayer *identityLayer = network->addIdentity(input);
    network->markOutput(identityLayer->getOutput(0));

    bool succ = builder->setMaxThreads(int32_t(5));
    if (succ) {
        std::cout << "set max threads success" << std::endl;
    } else {
        std::cout << "set max threads failed" << std::endl;
    }

    std::cout << "max threads = " << builder->getMaxThreads() << std::endl;
    if (builder->isNetworkSupported(network, config)) {
        std::cout << "network is suppored" << std::endl;
    } else {
        std::cout << "network is not supported" << std::endl;
    }
    if (builder->platformHasFastFp16()) {
        std::cout << "has fast fp16" << std::endl;
    } else {
        std::cout << "not have fast fp16" << std::endl;
    }
    if (builder->platformHasFastInt8()) {
        std::cout << "has fast int8" << std::endl;
    } else {
        std::cout << "not have fast int8" << std::endl;
    }
    if (builder->platformHasTf32()) {
        std::cout << "has tf32" << std::endl;
    } else {
        std::cout << "not have tf32" << std::endl;
    }

    IHostMemory *engineString = builder->buildSerializedNetwork(network, config)



    return 0;
}