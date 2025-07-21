# cdeeply-C
C/C++ interface to C Deeply's neural network generators

Using the library involves four steps:

0) Include `cdeeply_neural_network.h` and define a variable of type `CDNN` to hold the neural network.
1) Call `cdeeply_tabular_regressor(&myNN, ...)` or `cdeeply_tabular_encoder(&myNN, ...)` to train a neural network in supervised or unsupervised mode.  *This step requires an internet connection*! as the training is done server-side.
2) Call `run_CDNN(&myNN, ...)` as many times as you want to process new data samples -- one sample per function call.
3) Call `free_CDNN(&myNN)` when you're done.

**Function definitions:**

`errCode = cdeeply_tabular_regressor(&myNN, numInputs, numTargetOutputs, numSamples,`  
`        &trainingSamples, sampleTableTranspose, &outputRowOrColumnList, &importances,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth, maxActivationRate,`  
`        maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit,`  
`        allowedAFs, weightQuantization, activationQuantization,`  
`        weightSparsity, negativeWeights, ifNNhasBias, ifAllowingInputOutputConnections,`  
`        &sampleOutputs, &errorMessageString)`

Generates a x->y prediction network using *supervised* training on `trainingSamples`.
* `trainingSamples` is a `(numInputs+numTargetOutputs)*numSamples`-length table unrolled to type `double *`.
  * Set `sampleTableTranspose` to `FEATURE_SAMPLE_ARRAY` for `trainingSamples[input_output][sample]` array ordering, or `SAMPLE_FEATURE_ARRAY` for `trainingSamples[sample][input_output]` array ordering.
  * The rows/columns in `trainingSamples` corresponding to the target outputs are specified by `outputRowOrColumnList`.
* The optional `importances` argument weights the cost function of the target outputs.  Pass as a `numTargetOutputs*numSamples`-length table (ordered according to `sampleTableTranspose`) unrolled to type `double *`, or `NULL` if this parameter isn't being used.
* Optional parameters `maxWeights`, `maxHiddenNeurons` and `maxLayers` limit the size of the neural network, and `maxWeightDepth` limits the depth of layer-to-layer connections.  Set unused limits to `NO_MAX`.  The corresponding `maxWeightsHardLimit` and `maxHiddenNeuronsHardLimit` parameters should be either `HARD_LIMIT` or `SOFT_LIMIT`.
* `allowedAFs` is a parameter of type `AFlist`.  Each field of `allowedAFs` should be set to `ALLOWED_AF` or `OFF`, depending on whether the activation function should be considered for a given layer of the network.
* `weightQuantization` and `activationQuantization` are both parameters of the type `quantizationType`.  To quantize weights or neural activations, set the corresponding `ifQuantize` field to `QUANTIZE` and give values to the remaining three fields; otherwise set `ifQuantize` to `OFF`.
* Set `weightSparsity` to either `SPARSE_WEIGHTS` or `NONSPARSE_WEIGHTS`.
* The parameter `maxActivationRate` should be set to `1.` unless sparsifying the neural activations, in which case this can be a lower positive value.  Sparse activations do not work with sigmoid or tanh activation functions.  Set `maxActivationsHardLimit` to either `HARD_LIMIT` or `SOFT_LIMIT`.
* The `negativeWeights` parameter is either `NO_NEGATIVE_WEIGHTS` or `ALLOW_NEGATIVE_WEIGHTS`.  Set `ifNNhasBias` to `HAS_BIAS` unless you don't want to allow a bias (i.e. constant) term in each neuron's input, in which case set this to `NO_BIAS`.
* Set `ifAllowingInputOutputConnections` to `ALLOW_IO_CONNECTIONS` or `NO_IO_CONNECTIONS` depending on whether to allow the input layer to feed directly into the output layer.  (Outliers in new input data might cause wild outputs).
* `sampleOutputs` is an optional `numTargetOutputs*numSamples`-length table unrolled to type `double *`, to which the training output *as calculated by the server* will be written.  This is mainly a check that the data went through the pipes OK.  If you don't care about this parameter, set it to `NULL`.
* `errorMessageString` will point to the error message if something went wrong.  (The message should *not* be deallocated after being read).  Set to `NULL` if you don't care about the message.

`errCode = cdeeply_tabular_encoder(CDNN *myNN, numFeatures, numSamples,`  
`        &trainingSamples, sampleTableTranspose, &importances,`  
`        ifDoEncoder, ifDoDecoder, numEncodingFeatures, numVariationalFeatures, variationalDist,`  
`        maxWeights, maxHiddenNeurons, maxLayers, maxWeightDepth,`  
`        maxWeightsHardLimit, maxHiddenNeuronsHardLimit, maxActivationsHardLimit,`  
`        allowedAFs, weightQuantization, activationQuantization,`  
`        weightSparsity, negativeWeights, ifNNhasBias, &sampleOutputs, &errorMessageString)`

Generates an autoencoder (or an encoder or decoder) using *unsupervised* training on `trainingSamples`.
* `trainingSamples` is a `numFeatures*numSamples`-length table unrolled to type `double *`.
* `importances` and `sampleTableTranspose` are set the same way as for `cdeeply_tabular_regressor(...)`.
* The size of the encoding is determined by `numEncodingFeatures`.
  * So-called variational features are extra randomly-distributed inputs used by the decoder, analogous to the extra degrees of freedom a variational autoencoder generates.
  * `variationalDist` is set to `UNIFORM_DIST` if the variational inputs are uniformly-(0, 1)-distributed, or `NORMAL_DIST` if they are normally distributed (zero mean, unit variance).
* Set `ifDoEncoder` to either `DO_ENCODER` or `NO_ENCODER`, the latter being for a decoder-only network.
* Set `ifDoDecoder` to either `DO_DECODER` or `NO_DECODER`, the latter being for an encoder-only network.
* The remaining parameters are set the same way as for `cdeeply_tabular_regressor(...)`.

`oneSampleOutput = run_CDNN(&myNN, oneSampleInput)`

Runs the neural network on a *single* input sample, returning a pointer to the output of the network.  Note that this overwrites the last previously calculated network output.
* If it is a regressor or there are no variational features, `oneSampleInput` is a `double *` array having `numInputs` (regressor) or `numFeatures` ([auto]encoder) or `numEncodingFeatures` (decoder) elements.
* Any variational features should be generated randomly from the appropriate distribution, and appended to `oneSampleInput[]`, which now has `numFeatures+numVariationalFeatures` or `numEncodingFeatures+numVariationalFeatures` elements.
* The return value is simply the pointer to the last layer of the network, equivalent to `myNN.y[myNN.numLayers-1]`.

`void free_CDNN(CDNN *myNN)`

Frees memory associated with the neural network.

***

This library requires [libcurl](https://curl.se/libcurl/).  To compile the example using gcc, enter the command:

gcc cdeeply_neural_network.c CDNN_example.c -o CDNN_example -lm -lcurl
