/*
 *  example.c - interfaces to neural network generator
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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "cdeeply_neural_network.h"


double rand01()  {  return ((double) rand())/RAND_MAX;  }

int main(int argc, char **argv)
{
    CDNN NN;
    const int numFeatures = 10;
    const int numSamples = 100;
    int c2, io, f, s, rtrn, outputColumns[] = { numFeatures-1 };
    double trainTestMat[numFeatures*(numSamples+1)], *datum, dependentVar, featurePhase[numFeatures], featureCurvature[numFeatures];
    double *firstSampleOutputs, *testSampleOutputs, outputsComputedByServer[numFeatures*numSamples], targetValue;
    char *errMsg, *NNtypes[2] = { "autoencoder with 1 latent feature", "regressor" };
    char *targetDescriptions[2] = { "  Reconstructed test sample, feature 1", "  Test sample output" };
    
    srand((int) time(0));
 

        // generate a training matrix that traces out some noisy curve in Nf-dimensional space (noise ~ 0.1)
   
    for (f = 0; f < numFeatures; f++)  {
        featurePhase[f] = 2*M_PI*rand01();
        featureCurvature[f] = 2*M_PI*rand01();
    }
    datum = trainTestMat;
    for (s = 0; s < numSamples+1; s++)  {
        dependentVar = rand01();
        for (f = 0; f < numFeatures; f++)  {
            *datum = sin(featureCurvature[f]*dependentVar + featurePhase[f]) + 0.1*rand01();
            datum++;
    }   }
    
    
    for (c2 = 0; c2 < 2; c2++)  {
        
        printf("Generating %s..\n", NNtypes[c2]);
        if (c2 == 0)  {
            rtrn = cdeeply_tabular_encoder(&NN, numFeatures, numSamples, trainTestMat, SAMPLE_FEATURE_ARRAY, NULL,
                    DO_ENCODER, DO_DECODER, 1, 0, NORMAL_DIST,
                    NO_MAX, NO_MAX, NO_MAX, NO_MAX, HAS_BIAS, &outputsComputedByServer[0], &errMsg);      }
        else  {
            rtrn = cdeeply_tabular_regressor(&NN, numFeatures-1, 1, numSamples, trainTestMat, SAMPLE_FEATURE_ARRAY, outputColumns, NULL,
                    NO_MAX, NO_MAX, NO_MAX, NO_MAX, HAS_BIAS, ALLOW_IO_CONNECTIONS, &outputsComputedByServer[0], &errMsg);     }
        
        if (rtrn != 0)  {
            printf("  ** Server error %i (%s)\n", rtrn, errMsg);
            return rtrn;    }
        
        firstSampleOutputs = run_CDNN(&NN, trainTestMat);      // trainTestMat is also a pointer to sample #1's input vector
        if (fabs(firstSampleOutputs[0]-outputsComputedByServer[0]) > .0001)  {
            printf("  ** Network problem?  Sample 1 output was calculated as %g locally vs %g by the server\n", firstSampleOutputs[0], outputsComputedByServer[0]);
            return 200;    }
        
        testSampleOutputs = run_CDNN(&NN, trainTestMat + numSamples*numFeatures);    // this OVERWRITES firstSampleOutputs:  firstSampleOutputs == testSampleOutputs == NN.y[numLayers-1]
        if (c2 == 0)  targetValue = trainTestMat[numSamples*numFeatures];
        else  targetValue = trainTestMat[((numSamples+1)*numFeatures)-1];
        printf("%s was %g; target value was %g\n", targetDescriptions[c2], testSampleOutputs[0], targetValue);
        
        free_CDNN(&NN);
    }
    
    return 0;
}
