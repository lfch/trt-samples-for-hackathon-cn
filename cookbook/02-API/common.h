#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


using namespace nvinfer1;


#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}


class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

__inline__ std::string shapeToString(Dims32 dim)
{
    std::string output("[");
    if (dim.nbDims == 0)
    {
        return output + std::string("]");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string("]");
    return output;
}

__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}

size_t dataTypeToSize(DataType dataType) {
    switch (dataType) {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT8:
            return 1;
        case DataType::kINT32:
            return 4;
        case Dataype::kBOOL:
            return 1;
        case DataType::kUINT8:
            return 1;
        case DataType::kFP8:
            return 1;
        default:
            return 4;
    }
}

template <typename T>
void printArrayRecursively(const T *pArray, Dims32 dim, int iDim, int iStart) {
    if (iDim == dim.nbDims - 1) {
        for (int i = 0; i < dim.d[iDim]; ++i) {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << double(pArray[iStart + i]) << " ";
        }
    } else {
        int nElem = 1;
        for (int i = iDim + 1; i < dim.nbDims; ++i) {
            nElem *= dim.d[i];
        }
        for (int i = 0; i < dim.d[iDim]; ++i) {
            printArrayRecursively<T>(pArray, dim, iDim + 1, iStart + i * nElem)
        }
    }
    std::cout << std::endl;
}

template <typeName T>
void printArrayInfo(const T *pArray, Dims32 dim, std::string name = "", bool bPrintInfo = true, bool bPrintArrary = false, int n = 10){
    std::cout << std::endl;
    std::cout << name << ": (";
    for (int i = 0; i < dim.nbDims; i++) {
        std::cout << dim.d[i] << ", ";
    }
    std::cout << ")" << std::endl;

    // print information
    if (bPrintInfo) {
        int nElement = 1; // number of elements with batch dimension
        for (int i = 0; i < dim.nbDims; ++i)
        {
            nElement *= dim.d[i];
        }

        double sum      = double(pArray[0]);
        double absSum   = double(fabs(double(pArray[0])));
        double sum2     = double(pArray[0]) * double(pArray[0]);
        double diff     = 0.0;
        double maxValue = double(pArray[0]);
        double minValue = double(pArray[0]);
        for (int i = 1; i < nElement; ++i)
        {
            sum += double(pArray[i]);
            absSum += double(fabs(double(pArray[i])));
            sum2 += double(pArray[i]) * double(pArray[i]);
            maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
            minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
            diff += abs(double(pArray[i]) - double(pArray[i - 1]));
        }
        double mean = sum / nElement;
        double var  = sum2 / nElement - mean * mean;

        std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
        std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
        std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
        std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
        std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
        std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
        std::cout << std::endl;

               // print first n element and last n element
        for (int i = 0; i < n; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
        for (int i = nElement - n; i < nElement; ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
        }
        std::cout << std::endl;
    }    

    if (bPrintArrary) {
        printArrayRecursively<T>(pArray, dim, 0, 0);
    }
}

template void printArrayInfo(const float *, Dims32, std::string, bool, bool, int);
template void printArrayInfo(const half *, Dims32, std::string, bool, bool, int);
template void printArrayInfo(const char *, Dims32, std::string, bool, bool, int);
template void printArrayInfo(const int *, Dims32, std::string, bool, bool, int);
template void printArrayInfo(const bool *, Dims32, std::string, bool, bool, int);