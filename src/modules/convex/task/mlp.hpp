/* ----------------------------------------------------------------------- *//**
 *
 * @file mlp.hpp
 *
 * This file contains objective function related computation, which is called
 * by classes in algo/, e.g.,  loss, gradient functions
 *
 *//* ----------------------------------------------------------------------- */

#ifndef MADLIB_MODULES_CONVEX_TASK_MLP_HPP_
#define MADLIB_MODULES_CONVEX_TASK_MLP_HPP_

namespace madlib {

namespace modules {

namespace convex {

// Use Eigen
using namespace madlib::dbal::eigen_integration;

template <class Model, class Tuple>
class MLP {
public:
    typedef Model model_type;
    typedef Tuple tuple_type;
    typedef typename Tuple::independent_variables_type
        independent_variables_type;
    typedef typename Tuple::dependent_variable_type dependent_variable_type;

    static void gradientInPlace(
            model_type                          &model,
            const independent_variables_type    &y,
            const dependent_variable_type       &z,
            const double                        &stepsize);

    static double loss(
            const model_type                    &model,
            const independent_variables_type    &y,
            const dependent_variable_type       &z);

    static int predictClass(
            const model_type                    &model,
            const independent_variables_type    &y);

    static ColumnVector predict(
            const model_type                    &model,
            const independent_variables_type    &y);

    const static int RELU = 0;
    const static int SIGMOID = 1;
    const static int TANH = 2;

    static double sigmoid(const double &xi) {
        return 1. / (1. + std::exp(-xi));
    }

    static double relu(const double &xi) {
        return xi*(xi>0);
    }

    static double tanh(const double &xi) {
        return std::tanh(xi);
    }


private:

    static double sigmoidDerivative(const double &xi) {
        double value = sigmoid(xi);
        return value * (1. - value);
    }

    static double reluDerivative(const double &xi) {
        return xi>0;
    }

    static double tanhDerivative(const double &xi) {
        double value = tanh(xi);
        return 1-value*value;
    }

    static void feedForward(
            const model_type                    &model,
            const independent_variables_type    &y,
            std::vector<ColumnVector>           &net,
            std::vector<ColumnVector>           &x);

    static void endLayerDeltaError(
            const std::vector<ColumnVector>     &net,
            const std::vector<ColumnVector>     &x,
            const dependent_variable_type       &z,
            ColumnVector                        &delta_N);

    static void errorBackPropagation(
            const ColumnVector                  &delta_N,
            const std::vector<ColumnVector>     &net,
            const model_type                    &model,
            std::vector<ColumnVector>           &delta);
};

template <class Model, class Tuple>
void
MLP<Model, Tuple>::gradientInPlace(
        model_type                          &model,
        const independent_variables_type    &y,
        const dependent_variable_type       &z,
        const double                        &stepsize) {
    (void) model;
    (void) z;
    (void) y;
    (void) stepsize;
    std::vector<ColumnVector> net;
    std::vector<ColumnVector> x;
    std::vector<ColumnVector> delta;
    ColumnVector delta_N;

    feedForward(model, y, net, x);
    endLayerDeltaError(net, x, z, delta_N);
    errorBackPropagation(delta_N, net, model, delta);

    uint16_t N = model.u.size(); // assuming nu. of layers >= 1
    uint16_t k, s, j;
    // #TODO fix lambda
    float lambda = 0;

    std::vector<uint16_t> n; n.clear(); //nu. of units in each layer

    n.push_back(model.u[0].rows() - 1);
    for (k = 1; k <= N; k ++) {
        n.push_back(model.u[k-1].cols() - 1);
    }

    for (k=1; k <= N; k++){
        for (s=0; s <= n[k-1]; s++){
            for (j=1; j <= n[k]; j++){
                model.u[k-1](s,j) -= stepsize *  (delta[k](j) * x[k-1](s) + lambda * model.u[k-1](s,j));
            }
        }
    }
}

template <class Model, class Tuple>
double
MLP<Model, Tuple>::loss(
        const model_type                    &model,
        const independent_variables_type    &y,
        const dependent_variable_type       &z) {
    // Here we compute the loss. In the case of regression we use sum of square errors
    // In the case of classification the loss term is cross entropy.
    std::vector<ColumnVector> net;
    std::vector<ColumnVector> x;

    feedForward(model, y, net, x);
    double loss = 0.;
    uint16_t j;

    for (j = 1; j < z.rows() + 1; j ++) {
        if(model.is_classification){
            // Cross entropy: RHS term is negative
            loss -= z(j-1)*std::log(x.back()(j)) + (1-z(j-1))*std::log(1-x.back()(j));
            // Computed like https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/_base.py#L222
            //loss -= z(j-1)*std::log(x.back()(j));
        }else{
            double diff = x.back()(j) - z(j-1);
            loss += diff * diff;
        }
    }
    if(!model.is_classification){
        loss /= 2.;
    }else{
        // Give the average
        loss /= z.rows();
    }
    return loss;
}

template <class Model, class Tuple>
ColumnVector
MLP<Model, Tuple>::predict(
        const model_type                    &model,
        const independent_variables_type    &y) {
    (void) model;
    (void) y;
    std::vector<ColumnVector> net;
    std::vector<ColumnVector> x;

    feedForward(model, y, net, x);
    // Don't return the offset
    return x.back().tail(x.back().size()-1);
}

template <class Model, class Tuple>
int
MLP<Model, Tuple>::predictClass(
        const model_type                    &model,
        const independent_variables_type    &y) {
    (void) model;
    (void) y;
    std::vector<ColumnVector> net;
    std::vector<ColumnVector> x;

    feedForward(model, y, net, x);
    // Don't return the offset
    ColumnVector output = x.back().tail(x.back().size()-1);
    int max_idx;
    output.maxCoeff(&max_idx);
    return max_idx;
}

template <class Model, class Tuple>
void
// #TODO Change y to x and x to o
MLP<Model, Tuple>::feedForward(
        const model_type                    &model,
        const independent_variables_type    &y,
        std::vector<ColumnVector>           &net,
        std::vector<ColumnVector>           &x){
    // meta data and x_k^0 = 1
    uint16_t k, j, s;
    uint16_t N = model.u.size(); // assuming >= 1
    net.resize(N + 1);
    x.resize(N + 1);

    std::vector<uint16_t> n; n.clear();
    n.push_back(model.u[0].rows() - 1);
    x[0].resize(n[0] + 1);
    x[0](0) = 1.;
    for (k = 1; k <= N; k ++) {
        n.push_back(model.u[k-1].cols() - 1);
        net[k].resize(n[k] + 1);
        x[k].resize(n[k] + 1);
        // Bias
        x[k](0) = 1.;
    }

    // y is a mapped parameter from DB, aligning with x here
    for (j = 1; j <= n[0]; j ++) { x[0](j) = y(j-1); }

    for (k = 1; k < N; k ++) {
        for (j = 1; j <= n[k]; j ++) {
            net[k](j) = 0.;
            for (s = 0; s <= n[k-1]; s ++) {
                net[k](j) += x[k-1](s) * model.u[k-1](s, j);
            }
            if(model.activation==RELU)
                x[k](j) = relu(net[k](j));
            else if(model.activation==SIGMOID)
                x[k](j) = sigmoid(net[k](j));
            else if(model.activation==TANH)
                x[k](j) = tanh(net[k](j));
            else
                elog(WARNING,"Invalid activation");
        }
    }

    // output layer computation
    for (j = 1; j <= n[N]; j ++) {
        x[N](j) = 0.;
        for (s = 0; s <= n[N-1]; s ++) {
            x[N](j) += x[N-1](s) * model.u[N-1](s, j);
        }
    }
    // Numerically stable calculation of softmax
    ColumnVector last_x = x[N].tail(n[N]);
    if(model.is_classification){
        double max_x = last_x.maxCoeff();
        last_x = (last_x.array() - max_x).exp();
        last_x /= last_x.sum();
    }
    x[N].tail(n[N]) = last_x;
}

template <class Model, class Tuple>
void
MLP<Model, Tuple>::endLayerDeltaError(
        const std::vector<ColumnVector>     &net,
        const std::vector<ColumnVector>     &x,
        const dependent_variable_type       &z,
        ColumnVector                        &delta_N) {
    //meta data
    uint16_t t;
    uint16_t N = x.size() - 1; // assuming >= 1
    uint16_t n_N = x[N].rows() - 1;
    delta_N.resize(n_N + 1);

    for (t = 1; t <= n_N; t ++) {
		delta_N(t) = (x[N](t) - z(t-1));
    }
}

template <class Model, class Tuple>
void
MLP<Model, Tuple>::errorBackPropagation(
        const ColumnVector                  &delta_N,
        const std::vector<ColumnVector>     &net,
        const model_type                    &model,
        std::vector<ColumnVector>           &delta) {
    // meta data
    uint16_t k, j, t;
    uint16_t N = model.u.size(); // assuming >= 1
    delta.resize(N + 1);

    std::vector<uint16_t> n; n.clear();
    n.push_back(model.u[0].rows() - 1);
    for (k = 1; k <= N; k ++) {
        n.push_back(model.u[k-1].cols() - 1);
        delta[k].resize(n[k]+1);
    }
    delta[N] = delta_N;

    for (k = N - 1; k >= 1; k --) {
        for (j = 0; j <= n[k]; j ++) {
            delta[k](j) = 0.;
            for (t = 1; t <= n[k+1]; t ++) {
                delta[k](j) += delta[k+1](t) * model.u[k](j, t);
            }
            if(model.activation==RELU)
                delta[k](j) = delta[k](j) * reluDerivative(net[k](j));
            else if(model.activation==SIGMOID)
                delta[k](j) = delta[k](j) * sigmoidDerivative(net[k](j));
            else if(model.activation==TANH)
                delta[k](j) = delta[k](j) * tanhDerivative(net[k](j));
            else
                elog(WARNING,"Invalid activation");
        }
    }
}

} // namespace convex

} // namespace modules

} // namespace madlib

#endif

