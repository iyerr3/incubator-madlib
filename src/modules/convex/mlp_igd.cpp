/* ----------------------------------------------------------------------- *//**
 *
 * @file mlp_igd.cpp
 *
 * @brief Low-rank Matrix Factorization functions
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>
#include <modules/shared/HandleTraits.hpp>

#include "mlp_igd.hpp"

#include "task/mlp.hpp"
#include "algo/igd.hpp"
#include "algo/loss.hpp"

#include "type/tuple.hpp"
#include "type/model.hpp"
#include "type/state.hpp"

namespace madlib {

namespace modules {

namespace convex {

// These 2 classes contain public static methods that can be called
typedef IGD<MLPIGDState<MutableArrayHandle<double> >, MLPIGDState<ArrayHandle<double> >,
        MLP<MLPModel<MutableArrayHandle<double> >, MLPTuple > > MLPIGDAlgorithm;

typedef Loss<MLPIGDState<MutableArrayHandle<double> >, MLPIGDState<ArrayHandle<double> >,
        MLP<MLPModel<MutableArrayHandle<double> >, MLPTuple > > MLPLossAlgorithm;

/**
 * @brief Perform the multi-layer perceptron transition step
 *
 * Called for each tuple.
 */
AnyType
mlp_igd_transition::run(AnyType &args) {
    // For the first tuple: args[0] is nothing more than a marker that
    // indicates that we should do some initial operations.
    // For other tuples: args[0] holds the computation state until last tuple
    MLPIGDState<MutableArrayHandle<double> > state = args[0];

    // initilize the state if first tuple
    if (state.algo.numRows == 0) {
        if (!args[3].isNull()) {
            MLPIGDState<ArrayHandle<double> > previousState = args[3];

            state.allocate(*this, previousState.task.numberOfStages,
                           previousState.task.numbersOfUnits);
            state = previousState;
        } else {
            // configuration parameters
            ArrayHandle<int> numbersOfUnits = args[4].getAs<ArrayHandle<int> >();
            if (numbersOfUnits.size() <= 1) {
                throw std::runtime_error("Invalid parameter: numbers_of_units "
                        "has too few entries");
            } else if (numbersOfUnits.size() >=
                    std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("Invalid parameter: numbers_of_units "
                        "has too many entries");
            }
            size_t k;
            for (k = 0; k < numbersOfUnits.size(); k ++) {
                if (numbersOfUnits[k] == 0) {
                throw std::runtime_error("Invalid parameter: numbers_of_units "
                        "has zero entry");
                }
            }

            double stepsize = args[5].getAs<double>();
            if (stepsize <= 0.) {
                throw std::runtime_error("Invalid parameter: stepsize <= 0.0");
            }

            state.allocate(*this, numbersOfUnits.size() - 1,
                           reinterpret_cast<const int *>(numbersOfUnits.ptr()));
            state.task.stepsize = stepsize;


            int activation = args[6].getAs<int>();
            if (!( activation  == 0) &&  !( activation  == 1) && !(activation==2)) {
                throw std::runtime_error("Invalid parameter: activation should be 0, 1 or 2");
            }

            int is_classification = args[7].getAs<int>();
            if (!( is_classification  == 0) && !( is_classification  == 1)) {
                throw std::runtime_error("Invalid parameter: is_classification should be 0 or 1");
            }
            state.task.model.initialize(is_classification, activation);
        }

        // resetting in either case
        state.reset();
    }

    // meta data
    const uint16_t N = state.task.numberOfStages;
    const int *n = state.task.numbersOfUnits;

    // tuple
    MappedColumnVector indVar;
    MappedColumnVector depVar;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector x = args[1].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        indVar.rebind(x.memoryHandle(), x.size());
        MappedColumnVector y = args[2].getAs<MappedColumnVector>();
        depVar.rebind(y.memoryHandle(), y.size());

    } catch (const ArrayWithNullException &e) {
        return args[0];
    }
    MLPTuple tuple;
    tuple.indVar.rebind(indVar.memoryHandle(), indVar.size());
    tuple.depVar.rebind(depVar.memoryHandle(), depVar.size());
    //if (tuple.indVar.size() != n[0]){
        //throw std::runtime_error("Invalid parameter: input or output vector "
                //"does not have the correct numbers of elements");
    //}
    //if (tuple.depVar.size() != n[N]) {
        //throw std::runtime_error("Invalid parameter: input or output vector "
                //"does not have the correct numbers of elements");
    //}

    // Now do the transition step
    MLPIGDAlgorithm::transition(state, tuple);
    MLPLossAlgorithm::transition(state, tuple);
    state.algo.numRows ++;

    return state;
}

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
mlp_igd_merge::run(AnyType &args) {
    MLPIGDState<MutableArrayHandle<double> > stateLeft = args[0];
    MLPIGDState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.algo.numRows == 0) { return stateRight; }
    else if (stateRight.algo.numRows == 0) { return stateLeft; }

    // Merge states together
    MLPIGDAlgorithm::merge(stateLeft, stateRight);
    MLPLossAlgorithm::merge(stateLeft, stateRight);
    // The following numRows update, cannot be put above, because the model
    // averaging depends on their original values
    stateLeft.algo.numRows += stateRight.algo.numRows;

    return stateLeft;
}

/**
 * @brief Perform the multi-layer perceptron final step
 */
AnyType
mlp_igd_final::run(AnyType &args) {
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    MLPIGDState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.algo.numRows == 0) { return Null(); }

    // finalizing
    MLPIGDAlgorithm::final(state);

    // for stepsize tuning
    std::stringstream debug;
    debug << "loss: " << state.algo.loss << std::endl;
    warning(debug.str());
    return state;
}

/**
 * @brief Return the difference in RMSE between two states
 */
AnyType
internal_mlp_igd_distance::run(AnyType &args) {
    MLPIGDState<ArrayHandle<double> > stateLeft = args[0];
    MLPIGDState<ArrayHandle<double> > stateRight = args[1];
    return std::abs(stateLeft.algo.loss - stateRight.algo.loss);
}

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
internal_mlp_igd_result::run(AnyType &args) {
    MLPIGDState<ArrayHandle<double> > state = args[0];

    HandleTraits<ArrayHandle<double> >::ColumnVectorTransparentHandleMap
        flattenU;
    flattenU.rebind(&state.task.model.is_classification,&state.task.model.activation,&state.task.model.u[0](0, 0),
            state.task.model.arraySize(state.task.numberOfStages,
                    state.task.numbersOfUnits));
    double loss = state.algo.loss;

    AnyType tuple;
    tuple << flattenU << loss;
    return tuple;
}



AnyType internal_mlp_predict::run (AnyType& args)
{
    // returns NULL if the feature has NULL values
    try {
        args[0].getAs<MappedColumnVector>();
    } catch(const ArrayWithNullException &e) {
        return Null();
    }

    MappedColumnVector y = args[0].getAs<MappedColumnVector>();
    MappedColumnVector x_N;
    predict(model, y, x_N);
}

AnyType
mlp_predict_transition::run(AnyType &args) {
    MLPPredictState<MutableArrayHandle<double> > state = args[0];

    if (state.algo.numRows == 0) {
        MappedColumnVector coeff = args[1].getAs<MappedColumnVector>();
        MappedColumnVector layerSizes = args[2].getAs<MappedColumnVector>();
        size_t numberOfStages = numberOfStages.size();

        MLPModel model;
        model.rebind(is_classification,activation,coeff[0],numberOfStages,layerSizes[0])
    }

    MappedColumnVector indVar;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector x = args[1].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        indVar.rebind(x.memoryHandle(), x.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

}

} // namespace modules

} // namespace madlib

