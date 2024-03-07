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
#include "cdeeply.h"


int main(int argc, char **argv)
{
    CDNN NN;
    const int numInputs = 10;
    const int numSamples = 100;
    int c2, io, s, rtrn, outputColumns[] = { numInputs };
    double trainingData[(numInputs+1)*numSamples], *datum;
    double out1, outputCheck[numInputs*numSamples];
    char *errMsg, *NNtypes[2] = { "encoder", "regressor" };
    
    for (c2 = 0; c2 < 2; c2++)  {
        
        datum = trainingData;
        for (s = 0; s < numSamples; s++)  {
            double iSum = 0.;     // the last row is just some function
            for (io = 0; io < numInputs-c2; io++)  {
                *datum = ((double) rand())/RAND_MAX;
                iSum += (*datum) * sin((double) io) / numInputs;
                datum++;
            }
            *datum = cos(iSum);
            datum++;
        }
        
        printf("Generating %s..\n", NNtypes[c2]);
        if (c2 == 0)  {
            rtrn = cdeeply_tabular_encoder(&NN, numInputs, 1, 0, numSamples, trainingData, NULL, SAMPLE_FEATURE_ARRAY,
                    DO_ENCODER, DO_DECODER, NORMAL_DIST,
                    NO_MAX, NO_MAX, NO_MAX, NO_MAX, HAS_BIAS, &outputCheck[0], &errMsg);      }
        else  {
            rtrn = cdeeply_tabular_regressor(&NN, numInputs, 1, numSamples, trainingData, NULL, SAMPLE_FEATURE_ARRAY, outputColumns,
                    NO_MAX, NO_MAX, NO_MAX, NO_MAX, HAS_BIAS, ALLOW_IO_CONNECTIONS, &outputCheck[0], &errMsg);     }
        
        if (rtrn != 0)  {
            printf("  ** Server error %i (%s)\n", rtrn, errMsg);
            return rtrn;    }
        
        out1 = run_CDNN(&NN, trainingData)[0];      // trainingData is also a pointer to sample #1's input vector
        if (fabs(out1-outputCheck[0]) > .0001)  {
            printf("  ** Network problem?  Sample 1 output was calculated as %g locally vs %g by the server\n", out1, outputCheck[0]);
            return 200;    }
        
        printf("  Output on sample #1 was %g\n", NNtypes[c2], out1);
        
        free_CDNN(&NN);
    }
    
    return 0;
}
