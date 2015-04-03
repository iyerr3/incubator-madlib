/* ----------------------------------------------------------------------- *//**
 *
 * @file correlation.cpp
 *
 *//* ----------------------------------------------------------------------- */

#include <dbconnector/dbconnector.hpp>

#include "correlation.hpp"

namespace madlib {

namespace modules {

namespace stats {

using namespace dbal::eigen_integration;

// ----------------------------------------------------------------------

AnyType
correlation_transition::run(AnyType& args) {
    // args[2]
    if (args[2].isNull()) {
        throw std::runtime_error("Correlation: Mean vector is NULL.");
    }
    MappedColumnVector mean;
    try {
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        mean.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        throw std::runtime_error("Correlation: Mean vector contains NULL.");
    }
    // args[0]
    MutableNativeMatrix state;
    if (args[0].isNull()) {
        state.rebind(this->allocateArray<double>(mean.size(), mean.size()),
                     mean.size(), mean.size());
    } else {
        state.rebind(args[0].getAs<MutableArrayHandle<double> >());
    }
    // args[1]
    if (args[1].isNull()) { return state; }
    MappedColumnVector x;
    try {
        MappedColumnVector xx = args[1].getAs<MappedColumnVector>();
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return state;
    }
    state += (x - mean) * trans(x - mean);

    return state;
}

// ----------------------------------------------------------------------

AnyType
correlation_merge_states::run(AnyType& args) {
    if (args[0].isNull()) { return args[1]; }
    if (args[1].isNull()) { return args[0]; }

    MutableNativeMatrix state1 = args[0].getAs<MutableNativeMatrix>();
    MappedMatrix state2 = args[1].getAs<MappedMatrix>();

    triangularView<Upper>(state1) += state2;
    return state1;
}

// ----------------------------------------------------------------------

AnyType
correlation_final::run(AnyType& args) {
    MutableNativeMatrix covariance = args[0].getAs<MutableNativeMatrix>();

    Matrix denom(covariance.rows(), covariance.cols());
    ColumnVector sqrt_of_diag = covariance.diagonal().cwiseSqrt();
    triangularView<Upper>(denom) = sqrt_of_diag * trans(sqrt_of_diag);

    // correlation = cov(x, y) / (s(x) s(y))  , where s is std. deviation
    triangularView<Upper>(covariance) = covariance.cwiseQuotient(denom);

    // we explicitly set diagonal to one to eliminate precision issues
    covariance.diagonal().setOnes();
    return covariance;
}

} // stats

} // modules

} // madlib
