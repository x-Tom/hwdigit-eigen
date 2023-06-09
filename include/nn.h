#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <functional>


class NeuralNetwork {
public:
    std::vector<Eigen::MatrixXf> weights;
    std::vector<Eigen::MatrixXf> biases;
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> derivatives;
    // constexpr uint input_size = IMAGE_WIDTH*IMAGE_HEIGHT;
    uint input_size;
    Eigen::VectorXf inputs;
    std::vector<Eigen::VectorXf> layerOutputs;
    std::vector<Eigen::VectorXf> layerDerivatives;

    std::vector<float> losses;

    template <typename... Args>
    NeuralNetwork(int firstLayer, Args... restLayers) : input_size(firstLayer), inputs(Eigen::VectorXf::Zero(firstLayer)) {
        initLayers(firstLayer, restLayers...);
    }


    void initLayers(int firstLayer, std::pair<std::function<float(float)>, std::function<float(float)>> activationPair) {}
 
    template <typename... Args>
    void initLayers(int firstLayer, std::pair<std::function<float(float)>, std::function<float(float)>> firstActivationPair, int secondLayer, Args... restLayers) {
        
        layerOutputs.push_back(Eigen::VectorXf::Random(secondLayer));
        layerDerivatives.push_back(Eigen::VectorXf::Random(secondLayer));

        
        activations.push_back(firstActivationPair.first);
        derivatives.push_back(firstActivationPair.second);

        weights.push_back(Eigen::MatrixXf::Random(secondLayer, firstLayer));
        biases.push_back(Eigen::MatrixXf::Random(secondLayer, 1));


        if constexpr (sizeof...(restLayers) > 0) {
            initLayers(secondLayer, restLayers...);
        }
    }

    void printWeights(){
        uint i = 1;
        for(Eigen::MatrixXf wmat : weights){
            std::cout << "Layer (" << i << "):\n" << wmat.size() << "\n" << wmat << std::endl;
            i++;
        }
    }

    void printLayerOutputs(){
        uint l = 0;
        for (Eigen::VectorXf vec : layerOutputs){
            std::cout << "Layer (" << l << "):\n" << vec << std::endl << std::endl;
            l++;
        }
    }

    void printLayerDims(){
        std::cout << "Number of Layers: " << layerOutputs.size() << std::endl;
        std::cout << "Layer Sizes:" << std::endl;
        std::cout << "Input Layer: " << input_size << std::endl;
        uint l = 0;
        for (Eigen::VectorXf vec : layerOutputs){
            std::cout << "Layer (" << l << "): " << vec.size() << std::endl;
            // std::cout << "Layer weight size: "
            ++l;
        }
        l = 0;
        for(Eigen::MatrixXf wmat : weights){
            std::cout << "Layer weight size (" << l << "):" << wmat.size() << std::endl;
            ++l;
        }

    }


    enum {MEAN_SQUARE_ERROR, CROSS_ENTROPY_LOSS};


    // Feed-forward function
    Eigen::VectorXf feedForward(Eigen::VectorXf input, uint cost_function = CROSS_ENTROPY_LOSS) {
        if (input.size() != weights.at(0).cols()) {
            throw std::invalid_argument("Size of the input vector must match the size of the input layer.");
        }
        // layerOutputs.clear();
        // layerDerivatives.clear();
        inputs = input; // cache inputs
        Eigen::VectorXf output = input;
        Eigen::VectorXf deriv = Eigen::VectorXf::Zero(output.size());
        // std::cout << "Inputs:\n" << inputs << std::endl;
        for (size_t l = 0; l < weights.size(); ++l) {
            Eigen::VectorXf z = ((weights.at(l) * output).array() + biases.at(l).array()).matrix();
            
            // std::cout << "z^(" << l << "):\n " << z << std::endl;
            // std::cout << "softmax(z^(" << l << ")):\n " << z.unaryExpr(softmax) / z.unaryExpr(softmax).sum() << std::endl;

            // layerOutputs.push_back(input);
            if(cost_function == CROSS_ENTROPY_LOSS && l == weights.size()-1){
                // std::cout << "z^(" << l << "):\n " << z << std::endl;
                // std::cout << "sum(softmax(z^(" << l << "))):\n " << z.unaryExpr(exp).sum() << std::endl;

                output = (z.array() - z.maxCoeff()).exp();
                output = output / output.sum();

                deriv = Eigen::VectorXf::Ones(output.size());
                // layerOutputs.push_back(output);
                // layerDerivatives.push_back(deriv);
                layerOutputs.at(l) = output;
                layerDerivatives.at(l) = deriv;
            } else if(activations.at(l) != nullptr){ // 
                output = z.unaryExpr(activations.at(l));
                deriv = z.unaryExpr(derivatives.at(l));
                // if(output == z.unaryExpr(softmax)) { // softmax == std::exp. Only use softmax with Cross entropy loss
                //     output /= output.sum();
                //     // deriv /= deriv.sum()
                //     // layerDerivatives.push_back(input.unaryExpr(derivatives.at(l)));
                // }
                // layerOutputs.push_back(output);
                // layerDerivatives.push_back(deriv);
                layerOutputs.at(l) = output;
                layerDerivatives.at(l) = deriv;
                
            } // first activation function isnt used should be passed as nullptr
            // std::cout << "a^(" << l << "):\n " << output << std::endl;
        }
        // std::cout << "a^(L):\n " << layerOutputs.back() << std::endl;

        return layerOutputs.back();
    }


    std::pair<uint,Eigen::VectorXf> predict(Eigen::VectorXf input, uint cost_function = CROSS_ENTROPY_LOSS){
        Eigen::VectorXf probabilities = feedForward(input, cost_function);
        Eigen::VectorXf::Index max_index;
        probabilities.maxCoeff(&max_index);
        return std::make_pair(static_cast<uint>(max_index), probabilities);
    }

    std::pair<float,float> evaluate(std::vector<Eigen::VectorXf> x, std::vector<Eigen::VectorXf> y){
        if(x.size()!=y.size()) throw std::invalid_argument("No. Input samples not equal to No. of Labels");
        uint correct = 0;
        for(uint i = 0; i < x.size(); i++){
            std::pair<uint, Eigen::VectorXf> result = predict(x.at(i));
            uint predicted;
            Eigen::VectorXf probabilities;
            std::tie(predicted, probabilities) = result;
            if(i==x.size()-1) std::cout << probabilities << std::endl;
            // losses.push_back(cross_entropy_loss(probabilities, y.at(i)));
            Eigen::VectorXf::Index aidx;
            y[i].maxCoeff(&aidx);
            uint actual = static_cast<uint>(aidx);
            if(actual == predicted) correct++;
        }
        float accuracy = static_cast<float>(correct) / static_cast<float>(x.size());
        // float loss = stl_vec_mean(losses);
        float loss = 1;
        // std::vector<float>().swap(losses);
        return std::make_pair(accuracy,loss);
    }

    // Backpropagation
    void backpropagate_mse(const Eigen::VectorXf& desired, float learningRate) {
        // we choose cost function as sum(1/2(e)^2), where e = y - p, for simplicity
        // dc/da^(L) = e, L is last layer
        // 
        Eigen::VectorXf error = desired - layerOutputs.back(); // LayerOutputs.back() is output layer
        Eigen::VectorXf delta = error.array() * layerDerivatives.back().array();
        uint L = weights.size() - 1; // L is number of hidden layers and output layer - 1
        for (int l = L; l >= 0; --l) { // traverse back through layers
            if(l < L) delta = (weights.at(l+1).transpose() * delta).array() * layerDerivatives.at(l).array();
            Eigen::VectorXf output = (l > 0) ? layerOutputs.at(l - 1) : inputs; // da^(l)/dw^(l) = a^(l-1)
            weights.at(l) -= learningRate * delta * output.transpose();
            biases.at(l) -= learningRate * delta;
        }
    }

    void backpropagate_cel(const Eigen::VectorXf& desired, float learningRate) {
        // we choose cost function as sum(y*log(p)) and assume final layer uses softmax activation
        // dc/da^(L) = e, L is last layer
        // 
        Eigen::VectorXf error = layerOutputs.back() - desired; // LayerOutputs.back() is output layer
        // Eigen::VectorXf delta = error.array() * layerDerivatives.back().array();
        Eigen::VectorXf delta = error;
        uint L = weights.size() - 1; // L is number of hidden layers and output layer - 1
        for (int l = L; l >= 0; --l) { // traverse back through layers
            if(l < L) delta = (weights.at(l+1).transpose() * delta).array() * layerDerivatives.at(l).array();
            Eigen::VectorXf output = (l > 0) ? layerOutputs.at(l - 1) : inputs; // da^(l)/dw^(l) = a^(l-1)
            weights.at(l) -= learningRate * delta * output.transpose();
            biases.at(l) -= learningRate * delta;
        }
    }

    void backpropagate(const Eigen::VectorXf& desired, float learningRate, uint cost_function = MEAN_SQUARE_ERROR){
        switch(cost_function){
            case MEAN_SQUARE_ERROR:
                backpropagate_mse(desired, learningRate);
                break;
            case CROSS_ENTROPY_LOSS:
                backpropagate_cel(desired, learningRate);
                break;
        }
    } 

    float trainSGD(const std::vector<Eigen::VectorXf>& x, const std::vector<Eigen::VectorXf>& y, float learningRate, uint cost_function) {
        // std::assert(inputs.size() == outputs.size());
        if (x.size() != y.size()) {
            throw std::invalid_argument("Size of the samples vector must match the size of the labels vector.");
        }
        float correct = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            std::pair<uint, Eigen::VectorXf> result = predict(x.at(i));
            uint predicted;
            Eigen::VectorXf probabilities;
            std::tie(predicted, probabilities) = result;           
            // if(i%1000 && i!=0) std::cout << probabilities << std::endl;

            // if(!(i%5000) && i!=0) cross_entropy_loss(probabilities, y.at(i));

            Eigen::VectorXf::Index aidx;
            y.at(i).maxCoeff(&aidx);
            uint actual = static_cast<uint>(aidx);
            correct += (float)(predicted == actual);
            backpropagate(y.at(i), learningRate, cost_function);
            // float val_acc = nn.evaluate(x_val,y_val);
            // std::cout << "training data accuracy: " << val_acc << std::endl;
        }
        correct /= y.size();
        return correct;
    }   

    // Mini-batch training function 
    float trainMiniBatchGD(const std::vector<Eigen::VectorXf>& x, const std::vector<Eigen::VectorXf>& y, int batchSize, float learningRate) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("Size of the samples vector must match the size of the labels vector.");
        }
        std::vector<Eigen::MatrixXf> weightUpdates(weights.size());
        std::vector<Eigen::MatrixXf> biasUpdates(biases.size());

        float correct = 0;

        for (size_t i = 0; i < weightUpdates.size(); ++i) {
            weightUpdates.at(i) = Eigen::MatrixXf::Zero(weights.at(i).rows(), weights.at(i).cols());
            biasUpdates.at(i) = Eigen::MatrixXf::Zero(biases.at(i).rows(), biases.at(i).cols());
        }

        for (size_t i = 0; i < x.size(); ++i) {
            std::pair<uint, Eigen::VectorXf> result = predict(x.at(i));
            uint predicted;
            Eigen::VectorXf probabilities;
            std::tie(predicted, probabilities) = result;
            Eigen::VectorXf::Index aidx;
            y.at(i).maxCoeff(&aidx);
            uint actual = static_cast<uint>(aidx);
            correct += (float)(predicted == actual);

            Eigen::VectorXf error = layerOutputs.back() - y[i];
            Eigen::VectorXf delta = error;
            uint L = weights.size() - 1;
            for (int l = L; l >= 0; --l) {
                if(l < L) delta = (weights.at(l+1).transpose() * delta).array() * (layerDerivatives.at(l)).array();
                Eigen::VectorXf output = (l > 0) ? layerOutputs.at(l - 1) : inputs;
                weightUpdates.at(l) -= delta * output.transpose();
                biasUpdates.at(l) -= delta;
            }

            if (i % (batchSize-1) == 0 || i == x.size()-1) {
                for (size_t l = 0; l < weights.size(); ++l) {
                    weights.at(l) += learningRate * weightUpdates.at(l) / static_cast<float>(batchSize);
                    biases.at(l) += learningRate * biasUpdates.at(l) / static_cast<float>(batchSize);
                    weightUpdates.at(l).setZero();
                    biasUpdates.at(l).setZero();
                }
            }
        }
        return correct/x.size();
    }
};