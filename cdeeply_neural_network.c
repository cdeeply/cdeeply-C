/*
 *  cdeeply_neural_network.c - interfaces to neural network generator
 *  
 *  C Deeply
 *  Copyright (C) 2023 C Deeply, LLC
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

/*
 *  usage:
 *  
 *  1) Generate a neural network, using either of:
 *  
 *  int errCode = myNN.tabular_regressor( CDNN *myNN, int numInputs, int numOutputs, int numSamples, double *trainingSamples,
 *                FEATURE_SAMPLE_ARRAY or SAMPLE_FEATURE_ARRAY, int *outputIndices, double *importances or NULL,
 *                int maxWeights or NO_MAX, int maxHiddenNeurons or NO_MAX, int maxLayers or NO_MAX, int maxLayerSkips or NO_MAX,
 *                NO_BIAS or HAS_BIAS, NO_IO_CONNECTIONS or ALLOW_IO_CONNECTIONS, double *trainingOutputs or NULL, char **errorMessage or NULL );
 *  
 *  int errCode = myNN.tabular_encoder( CDNN *myNN, int numFeatures, int numSamples,
 *                double *trainingSamples, FEATURE_SAMPLE_ARRAY or SAMPLE_FEATURE_ARRAY, double *importances or NULL,
 *                DO_ENCODER or NO_ENCODER, DO_DECODER or NO_DECODER,
 *                int numEncodingFeatures, int numVariationalFeatures, NORMAL_DIST or UNIFORM_DIST,
 *                int maxWeights or NO_MAX, int maxHiddenNeurons or NO_MAX, int maxLayers or NO_MAX, int maxLayerSkips or NO_MAX,
 *                NO_BIAS or HAS_BIAS, double *trainingOutputs or NULL, char **errorMessage or NULL );
 *  
 *  * Pass "SAMPLE_FEATURE_ARRAY" if elements of trainingSamples are ordered (s1f1, s1f2, ..., s2f1, ...),
 *        or "FEATURE_SAMPLE_ARRAY" if elements of trainingSamples are ordered (s1f1, s2f1, ..., s1f2, ...).
 *  * For supervised x->y regression, the sample table contains BOTH 'x' and 'y', the latter specified by outputIndices[].
 *  * The importances table, if not NULL, has numOutputFeatures*numSamples elements (again, ordered by indexOrder),
 *        and weights the training cost function:  C = sum(Imp*dy^2).
 *  * Weight/neuron/etc limits are either positive integers or "NO_MAX".
 *  * trainingOutputs[], if passed, has numOutputs*numSamples elements and should agree with what's computed locally.
 *  * errorMessage, if passed, does not allocate a string and therefore does not need to be freed if it is set (i.e. if errCode != 0).
 *  
 *  
 *  2) Run the network on a (single) new sample
 *  
 *  double *oneSampleOutput = run_CDNN(CDNN *myNN, double *oneSampleInput);
 *  
 *  where oneSampleInput is a list of length numInputFeatures, and oneSampleOutput is a list of length numOutputFeatures.
 *  * If it's an autoencoder (encoder+decoder), length(oneSampleInput) and length(oneSampleOutput) equal the size of the training sample space.
 *        If it's just an encoder, length(oneSampleOutput) equals numEncodingFeatures; if decoder only, length(oneSampleInput) must equal numEncodingFeatures.
 *    If it's a decoder/autoencoder network having numVariationalFeatures > 0, then oneSampleInput has numEncodingFeatures/numInputFeatures sample inputs
 *        followed by numVariationalFeatures random numbers drawn from variationalDistribution.
 *  
 *  
 *  3) Free memory
 * 
 *  free_CDNN(CDNN *myNN);
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cdeeply_neural_network.h"
#include <curl/curl.h>


typedef struct { const char *name; char **data; } postField;
typedef struct { void *nextBuffer; int numChars; } bufferType;

char *charPtr, *endChars;
long curlWriteDataSize;
bufferType *readBuffer;


int cdReadNum(void *theNum, int mode)
{
    char *numEnd = charPtr;
    while ((*numEnd != ',') && (*numEnd != ';'))  {
        numEnd++;
        if (numEnd > endChars)  return CD_NN_READ_ERROR;   }
    *numEnd = 0;
    
    if (mode == 0)  sscanf(charPtr, "%i", (int *) theNum);
    else  sscanf(charPtr, "%lg", (double *) theNum);
    
    charPtr = numEnd+1;
    
    return 0;
}

int cdReadInt(int *theInt)  {  return cdReadNum((void *) theInt, 0);  }
int cdReadInts(int *theInts, int numInts)
{
    int i;
    for (i = 0; i < numInts; i++)  {
    if (cdReadInt(theInts + i) != 0)  {
        return CD_NN_READ_ERROR;
    }}
    return 0;
}

int cdReadFloat(double *theFloat)  {  return cdReadNum((void *) theFloat, 1);  }
int cdReadFloats(double *theFloats, int numFloats)
{
    int i;
    for (i = 0; i < numFloats; i++)  {
    if (cdReadFloat(theFloats + i) != 0)  {
        return CD_NN_READ_ERROR;
    }}
    return 0;
}


char *data2table(double *data, int numIOs, int numSamples, int indexOrder)
{
    int dim1, dim2, i1, i2, numLength;
    char *table, *charPtr;
    double *numPtr;
    
    table = malloc(25*numIOs*numSamples);
    if (table == NULL)  return table;
    
    if (indexOrder == FEATURE_SAMPLE_ARRAY)  {  dim1 = numIOs; dim2 = numSamples;  }
    else  {  dim1 = numSamples; dim2 = numIOs;  }
    
    charPtr = table;
    numPtr = data;
    for (i1 = 0; i1 < dim1; i1++)  {
        if (i1 > 0)  {  *charPtr = '\n';  charPtr++;  }
        for (i2 = 0; i2 < dim2; i2++)  {
            if (i2 > 0)  {  *charPtr = ',';  charPtr++;  }
            sprintf(charPtr, "%g%n", *numPtr, &numLength);
            charPtr += numLength;
            numPtr++;
    }   }
    *charPtr = 0;
    
    return table;
}



int cdeeply_getNN(CDNN *NN, char *NNchars, long numChars, double *sampleOutputs, int numSamples)
{
    char header[100];
    int loopChar, rtrn, l, li, inputLayer, numWeights;
    
    NNchars[numChars] = ';';
    
    charPtr = NNchars;
    endChars = charPtr+numChars;
    
    if (cdReadInt(&NN->numLayers) != 0)  return CD_NN_READ_ERROR;
    if (cdReadInt(&NN->encoderLayer) != 0)  return CD_NN_READ_ERROR;
    if (cdReadInt(&NN->variationalLayer) != 0)  return CD_NN_READ_ERROR;
    
    NN->layerSize = malloc(NN->numLayers*sizeof(int));
    NN->layerAFs = malloc(NN->numLayers*sizeof(int));
    NN->numLayerInputs = malloc(NN->numLayers*sizeof(int));
    NN->layerInputs = malloc(NN->numLayers*sizeof(int *));
    NN->weights = malloc(NN->numLayers*sizeof(double **));
    NN->y = malloc(NN->numLayers*sizeof(double *));
    if ((NN->layerSize == NULL) || (NN->layerAFs == NULL) || (NN->numLayerInputs == NULL)
            || (NN->layerInputs == NULL) || (NN->weights == NULL) || (NN->y == NULL))
        return CD_OUT_OF_MEMORY_ERROR;
    
    if (cdReadInts(NN->layerSize, NN->numLayers) != 0)  return CD_NN_READ_ERROR;
    if (cdReadInts(NN->layerAFs, NN->numLayers) != 0)  return CD_NN_READ_ERROR;
    if (cdReadInts(NN->numLayerInputs, NN->numLayers) != 0)  return CD_NN_READ_ERROR;
    
    for (l = 0; l < NN->numLayers; l++)  {
        NN->layerInputs[l] = malloc(NN->numLayerInputs[l]*sizeof(int));
        NN->y[l] = malloc(NN->layerSize[l]*sizeof(double));
        if ((NN->layerInputs[l] == NULL) || (NN->y[l] == NULL))  return CD_OUT_OF_MEMORY_ERROR;
        
        if (cdReadInts(NN->layerInputs[l], NN->numLayerInputs[l]) != 0)  return CD_NN_READ_ERROR;
    }
    for (l = 0; l < NN->numLayers; l++)  {
        NN->weights[l] = malloc(NN->numLayerInputs[l]*sizeof(double *));
        if (NN->weights[l] == NULL)  return CD_OUT_OF_MEMORY_ERROR;
        
        for (li = 0; li < NN->numLayerInputs[l]; li++)  {
            inputLayer = NN->layerInputs[l][li];
            numWeights = NN->layerSize[l]*NN->layerSize[inputLayer];
            
            NN->weights[l][li] = malloc(numWeights*sizeof(double));
            if (NN->weights[l][li] == NULL)  return CD_OUT_OF_MEMORY_ERROR;
            
            if (cdReadFloats(NN->weights[l][li], numWeights) != 0)  return CD_NN_READ_ERROR;
    }   }
    
    if (sampleOutputs != NULL)  {
    if (cdReadFloats(sampleOutputs, NN->layerSize[NN->numLayers-1]*numSamples) != 0)  {
        return CD_NN_READ_ERROR;
    }}
    
    return 0;
}



static size_t curlWriteCallback(void *newData, size_t typeSize, size_t dataLength, void *ptr)
{
    size_t numBytes = dataLength*typeSize;
    bufferType *newBuffer;
    
    newBuffer = (bufferType *) malloc(sizeof(bufferType) + numBytes);
    if (newBuffer == NULL)  return 0;
    
    newBuffer->nextBuffer = NULL;
    newBuffer->numChars = numBytes;
    memcpy((void *) (newBuffer + 1), newData, numBytes);
    curlWriteDataSize += numBytes;
    
    if (*(bufferType **) ptr == NULL)  *(bufferType **) ptr = newBuffer;
    else  readBuffer->nextBuffer = (bufferType *) newBuffer;
    readBuffer = newBuffer;
    
    return numBytes;
}


char errMsgChars[400];

void setErrMsg(char *msg)
{
    int i;
    char *c = errMsgChars;
    
    for (i = 0; i < sizeof(errMsgChars)-4; i++)  {
        *c = msg[i];
        if (*c == 0)  return;
        c++;
    }
    
    c[0] = c[1] = c[2] = '.';
    c[3] = 0;
}

int cdeeply_buildNN(CDNN *NN, double *sampleOutputs,
            int numSamples, postField *toPOST, int numPostFields)
{
    int p, rtrn;
    bufferType *readBuffer1;
    char *readBufferChars, *readBufferScanner;
    CURL *curlP;
    curl_mime *mime;
    curl_mimepart *mimePart;
    
    curl_global_init(CURL_GLOBAL_ALL);
    curlP = curl_easy_init();
    if (curlP != NULL)  mime = curl_mime_init(curlP);
    if ((curlP == NULL) || (mime == NULL))  return CD_OUT_OF_MEMORY_ERROR;
    
    curl_easy_setopt(curlP, CURLOPT_URL, "https://cdeeply.com/myNN.php");
    
    for (p = 0; p < numPostFields; p++)  {
        mimePart = curl_mime_addpart(mime);
        if (mimePart == NULL)  return CD_OUT_OF_MEMORY_ERROR;
        rtrn = curl_mime_name(mimePart, toPOST[p].name);
        if (rtrn != CURLE_OK)  return rtrn;
        rtrn = curl_mime_data(mimePart, *toPOST[p].data, CURL_ZERO_TERMINATED);
        if (rtrn != CURLE_OK)  return rtrn;
    }
    curl_easy_setopt(curlP, CURLOPT_MIMEPOST, mime);
    
    readBuffer1 = NULL;
    curlWriteDataSize = 0;
    curl_easy_setopt(curlP, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curlP, CURLOPT_WRITEDATA, (void *) &readBuffer1);
    
    rtrn = curl_easy_perform(curlP);
    if (rtrn != CURLE_OK)  return rtrn;
    
    curl_mime_free(mime);
    
    readBufferScanner = readBufferChars = malloc(curlWriteDataSize+1);
    if (readBufferChars == NULL)  return CD_OUT_OF_MEMORY_ERROR;
    readBuffer = readBuffer1;
    while (readBuffer != NULL)  {
        memcpy(readBufferScanner, readBuffer+1, readBuffer->numChars);
        readBufferScanner += readBuffer->numChars;
        readBuffer1 = readBuffer;
        readBuffer = (bufferType *) readBuffer->nextBuffer;
        free(readBuffer1);
    }
    *readBufferScanner = 0;
    
    if (!((*readBufferChars >= '0') && (*readBufferChars <= '9')))  {
        setErrMsg(readBufferChars);
        rtrn = CD_PARAMS_ERR;
    }
    else  {
        rtrn = cdeeply_getNN(NN, readBufferChars, curlWriteDataSize, sampleOutputs, numSamples);
        if (rtrn == CD_NN_READ_ERROR)  setErrMsg("Problem reading neural network from server");
        else if (rtrn == CD_NN_READ_ERROR)  setErrMsg(readBufferChars);
    }
    
    free(readBufferChars);
    curl_easy_cleanup(curlP);
    curl_global_cleanup();
    
    return rtrn;
}


char *checked[3] = { "", "on", "off" };
char *vDists[2] = { "uniform", "normal" };
char *NNtypes[2] = { "autoencoder", "regressor" };
char *SubmitStr = "Submit";
char *sourceStr = "C_API";

int cdeeply_tabular_encoder(CDNN *NN, int numFeatures, int numSamples,
        double *trainingSamples, int indexOrder, double *importances,
        int doEncoder, int doDecoder, int numEncodingFeatures, int numVariationalFeatures, int variationalDist,
        int maxWeights, int maxNeurons, int maxLayers, int maxLayerSkips,
        int hasBias, double *sampleOutputs, char **errMsg)
{
    int rtrn;
    char *trainingSamplesStr, *trainingSampleImportancesStr, strs[120], *numEncodingFeaturesStr = &strs[0];
    char *numVFsStr = &strs[20], *maxWeightsStr = &strs[40], *maxNeuronsStr = &strs[60];
    char *maxLayersStr = &strs[80], *maxLayerSkipsStr = &strs[100];
    char *rowcol[2] = { "columns", "rows" };
    postField toPOST[] = {
        { "samples", &trainingSamplesStr },
        { "importances", &trainingSampleImportancesStr },
        { "rowscols", &rowcol[indexOrder] },
        { "numFeatures", &numEncodingFeaturesStr },
        { "doEncoder", &checked[doEncoder] },
        { "doDecoder", &checked[doDecoder] },
        { "numVPs", &numVFsStr },
        { "variationalDist", &vDists[variationalDist] },
        { "maxWeights", &maxWeightsStr },
        { "maxNeurons", &maxNeuronsStr },
        { "maxLayers", &maxLayersStr },
        { "maxSkips", &maxLayerSkipsStr },
        { "hasBias", &checked[hasBias] },
        { "submitStatus", &SubmitStr },
        { "NNtype", &NNtypes[0] },
        { "formSource", &sourceStr }
    };
    
    trainingSamplesStr = data2table(trainingSamples, numFeatures, numSamples, indexOrder);
    if (importances == NULL)  trainingSampleImportancesStr = "";
    else  trainingSampleImportancesStr = data2table(importances, numFeatures, numSamples, indexOrder);
    if ((trainingSamplesStr == NULL) || (trainingSampleImportancesStr == NULL)) return CD_OUT_OF_MEMORY_ERROR;
    
    sprintf(numEncodingFeaturesStr, "%i", numEncodingFeatures);
    sprintf(numVFsStr, "%i", numVariationalFeatures);
    maxWeightsStr[0] = maxNeuronsStr[0] = maxLayersStr[0] = maxLayerSkipsStr[0] = 0;
    if (maxWeights >= 0)  sprintf(maxWeightsStr, "%i", maxWeights);
    if (maxNeurons >= 0)  sprintf(maxNeuronsStr, "%i", maxNeurons);
    if (maxLayers >= 0)  sprintf(maxLayersStr, "%i", maxLayers);
    if (maxLayerSkips >= 0)  sprintf(maxLayerSkipsStr, "%i", maxLayerSkips);
    
    rtrn = cdeeply_buildNN(NN, sampleOutputs, numSamples, toPOST, sizeof(toPOST)/sizeof(postField));
    if (errMsg != NULL)  *errMsg = &errMsgChars[0];
    
    free(trainingSamplesStr);
    if (importances != NULL)  free(trainingSampleImportancesStr);
    
    return rtrn;
}


int cdeeply_tabular_regressor(CDNN *NN, int numInputs, int numOutputs, int numSamples,
        double *trainingSamples, int indexOrder, int *outputRowsColumns, double *importances,
        int maxWeights, int maxNeurons, int maxLayers, int maxLayerSkips,
        int hasBias, int allowIOconnections, double *sampleOutputs, char **errMsg)
{
    int o, charIdx, rtrn;
    char *outputRowsColsStr, *trainingSamplesStr, *trainingSampleImportancesStr, strs[80], *maxWeightsStr = &strs[0];
    char *maxNeuronsStr = &strs[20], *maxLayersStr = &strs[40], *maxLayerSkipsStr = &strs[60];
    char *rowcol[2] = { "rows", "columns" };
    postField toPOST[] = {
        { "samples", &trainingSamplesStr },
        { "importances", &trainingSampleImportancesStr },
        { "rowscols", &rowcol[indexOrder] },
        { "rowcolRange", &outputRowsColsStr },
        { "maxWeights", &maxWeightsStr },
        { "maxNeurons", &maxNeuronsStr },
        { "maxLayers", &maxLayersStr },
        { "maxSkips", &maxLayerSkipsStr },
        { "hasBias", &checked[hasBias] },
        { "allowIO", &checked[allowIOconnections] },
        { "submitStatus", &SubmitStr },
        { "NNtype", &NNtypes[1] },
        { "formSource", &sourceStr }
    };
    
    trainingSamplesStr = data2table(trainingSamples, numInputs+numOutputs, numSamples, indexOrder);
    if (importances == NULL)  trainingSampleImportancesStr = "";
    else  trainingSampleImportancesStr = data2table(importances, numOutputs, numSamples, indexOrder);
    if ((trainingSamplesStr == NULL) || (trainingSampleImportancesStr == NULL)) return CD_OUT_OF_MEMORY_ERROR;
    
    outputRowsColsStr = malloc(numOutputs*20*sizeof(int));
    if (outputRowsColsStr == NULL)  return CD_OUT_OF_MEMORY_ERROR;
    
    if (numOutputs <= 0)  {  charIdx = 0; outputRowsColsStr[0] = 0;  }
    else  {
        charIdx = sprintf(outputRowsColsStr, "%i", outputRowsColumns[0]+1);
        for (o = 1; o < numOutputs; o++)  {
            charIdx += sprintf(outputRowsColsStr + charIdx, ",%i", outputRowsColumns[o]+1);
    }   }
    
    maxWeightsStr[0] = maxNeuronsStr[0] = maxLayersStr[0] = maxLayerSkipsStr[0] = 0;
    if (maxWeights >= 0)  sprintf((char *) maxWeightsStr, "%i", maxWeights);
    if (maxNeurons >= 0)  sprintf(maxNeuronsStr, "%i", maxNeurons);
    if (maxLayers >= 0)  sprintf(maxLayersStr, "%i", maxLayers);
    if (maxLayerSkips >= 0)  sprintf(maxLayerSkipsStr, "%i", maxLayerSkips);
    
    rtrn = cdeeply_buildNN(NN, sampleOutputs, numSamples,
            toPOST, sizeof(toPOST)/sizeof(postField));
    if (errMsg != NULL)  *errMsg = &errMsgChars[0];
    
    free(outputRowsColsStr);
    free(trainingSamplesStr);
    if (importances != NULL)  free(trainingSampleImportancesStr);
    
    return rtrn;
}



double linearAF(const double x)  {  return x;  }

double (*fs[5])(const double) = { &linearAF, NULL, NULL, NULL, &tanh };

double *run_CDNN(CDNN *NN, double *inputs)
{
    int l, li, l0, n, i, i0;
    double *w;
    
    NN->y[0][0] = 1;
    memcpy(NN->y[1], inputs, NN->layerSize[1]*sizeof(double));
    if (NN->variationalLayer > 0)  memcpy(NN->y[NN->variationalLayer],
            inputs+NN->layerSize[1], NN->layerSize[NN->variationalLayer]*sizeof(double));
    
    for (l = 2; l < NN->numLayers; l++)  {
    if (l != NN->variationalLayer)  {
        for (n = 0; n < NN->layerSize[l]; n++)  NN->y[l][n] = 0.;
        for (li = 0; li < NN->numLayerInputs[l]; li++)  {
            l0 = NN->layerInputs[l][li];
            w = NN->weights[l][li];
            for (i = 0; i < NN->layerSize[l]; i++)  {
            for (i0 = 0; i0 < NN->layerSize[l0]; i0++)  {
                NN->y[l][i] += (*w) * NN->y[l0][i0];
                w++;
        }   }}
        for (n = 0; n < NN->layerSize[l]; n++)  {
            NN->y[l][n] = fs[NN->layerAFs[l]](NN->y[l][n]);
    }}  }
    
    return NN->y[NN->numLayers-1];
}


void free_CDNN(CDNN *NN)
{
    int l, li;
    
    for (l = 0; l < NN->numLayers; l++)  {
        for (li = 0; li < NN->numLayerInputs[l]; li++)  {
            free(NN->weights[l][li]);       }
        
        free(NN->layerInputs[l]);
        free(NN->weights[l]);
        free(NN->y[l]);
    }
    
    free(NN->layerSize);
    free(NN->layerAFs);
    free(NN->numLayerInputs);
    free(NN->layerInputs);
    free(NN->weights);
    free(NN->y);
}
