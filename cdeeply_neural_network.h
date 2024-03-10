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


extern int CDNN_tabular_regressor(CDNN *, int, int, int, double *, int, int *, double *,
        int, int, int, int, int, int, double *, char **);
extern int CDNN_tabular_encoder(CDNN *, int, int, double *, int, double *,
        int, int, int, int, int, int, int, int, int, int, double *, char **);
extern double *run_CDNN(CDNN *, double *);
extern void free_CDNN(CDNN *);


#ifdef __cplusplus
}
#endif

#endif

