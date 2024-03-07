/*
 *  cdeeply_neural_network.h - interfaces to neural network generator
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

#ifndef cdeeply_h
#define cdeeply_h

#ifdef __cplusplus
extern "C" {
#endif


// Arguments to cdeeply_tabular_encoder() and cdeeply_tabular_regressor()

#define FEATURE_SAMPLE_ARRAY 0
#define SAMPLE_FEATURE_ARRAY 1

#define NO_ENCODER 0
#define DO_ENCODER 1
#define NO_DECODER 0
#define DO_DECODER 1

#define UNIFORM_DIST 0
#define NORMAL_DIST 1

#define NO_BIAS 0
#define HAS_BIAS 1

#define NO_IO_CONNECTIONS 0
#define ALLOW_IO_CONNECTIONS 1

#define NO_MAX -1


// Error codes, not including the error codes that libcurl returns

#define CD_OUT_OF_MEMORY_ERROR 100
#define CD_PARAMS_ERR 101
#define CD_NN_READ_ERROR 102

typedef struct {
    int numLayers, encoderLayer, variationalLayer;
    int *layerSize, *layerAFs, *numLayerInputs;
    int **layerInputs;
    double ***weights;
    double **y;
} CDNN;


extern int cdeeply_tabular_regressor(CDNN *, int, int, int, double *, int, int *, double *,
        int, int, int, int, int, int, double *, char **);
extern int cdeeply_tabular_encoder(CDNN *, int, int, double *, int, double *,
        int, int, int, int, int, int, int, int, int, int, double *, char **);
extern double *run_CDNN(CDNN *, double *);
extern void free_CDNN(CDNN *);


#ifdef __cplusplus
}
#endif

#endif

