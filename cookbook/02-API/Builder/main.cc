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

    


    return 0;
}