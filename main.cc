#include <iostream>
#include <cstring>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "grid_search.h"
#include "math_functions.h"
#include "data.h"
#include "nn.h"

constexpr uint FONT_SIZE = 22;
constexpr uint FONT_SIZE_SMALL = 12;

// constexpr uint FONT_SIZE = 24;

// constexpr float LEARNING_RATE = 2e-5f;
// constexpr float LEARNING_RATE = 2e-2f;
constexpr float LEARNING_RATE = 1e-1f;

// constexpr unsigned EPOCHS = 0;

constexpr unsigned EPOCHS = 100;
// constexpr unsigned BATCH_SIZE = 256;
constexpr unsigned BATCH_SIZE = 64;

using namespace std::chrono_literals;


int main() {
    using NN = NeuralNetwork;

    std::random_device rd;
    std::mt19937 rng(rd());

    NeuralNetwork nn(
        784, // INPUT LAYER
        std::make_pair(relu, relu_derivative), 16, 
        std::make_pair(relu, relu_derivative), 16, 
        std::make_pair(relu, relu_derivative), 10
    );

    // std::cout << nn.nnInfoString() << std::endl;

    std::cout << "Loading training data..." << std::endl;

    std::vector<pixel_vector> x_test = read_mnist_pixel_data("x_test.txt", 10000);
    std::vector<label_vector> y_test = read_mnist_label_data("y_test.txt", 10000);

    std::vector<pixel_vector> x_data = read_mnist_pixel_data("x_train.txt", 60000);
    std::vector<label_vector> y_data = read_mnist_label_data("y_train.txt", 60000);
    std::vector<pixel_vector> x_train(x_data.begin(), x_data.begin() + 50000);
    std::vector<label_vector> y_train(y_data.begin(), y_data.begin() + 50000);
    std::vector<pixel_vector> x_val(x_data.begin()+50000, x_data.end());
    std::vector<label_vector> y_val(y_data.begin()+50000, y_data.end());
    std::vector<pixel_vector>().swap(x_data);
    std::vector<label_vector>().swap(y_data);

    std::cout << "Loaded training data." << std::endl;
    
    nn.printLayerDims();
    
    for(int e = 0; e < EPOCHS; e++){
        // float train_accuracy = nn.trainSGD(x_train, y_train, LEARNING_RATE, NN::CROSS_ENTROPY_LOSS);
        float train_accuracy = nn.trainMiniBatchGD(x_train, y_train, BATCH_SIZE, LEARNING_RATE);
        auto [val_accuracy,_] = nn.evaluate(x_val, y_val);
        std::cout << "Epoch " << e+1 << " completed!" << std::endl;
        std::cout << "Training Accuracy: " << train_accuracy << std::endl;
        std::cout << "Validation Accuracy: " << val_accuracy << std::endl;
        two_vec_shuffle<pixel_vector, label_vector>(x_train, y_train, rng);
    }

    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_Texture *texture = nullptr;
    TTF_Font *font = nullptr;
    TTF_Font *font_small = nullptr;
    
    // SDL_Surface *text_surface = nullptr;
    // SDL_Texture *text_texture = nullptr;
    SDL_Color textColor = { 255, 255, 255, 0 }; // white
    SDL_Color textGreen = {   0, 255,   0, 0 };
    SDL_Color textRed =   { 255,   0,   0, 0 };



    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Could not initialize sdl2: %s\n", SDL_GetError());
        return 1;
    }

    if (TTF_Init() == -1) {
        printf("TTF_Init: %s\n", TTF_GetError());
        return 2;
    }

    window = SDL_CreateWindow("Handwritten Digit Classifier", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        printf("Could not create renderer: %s\n", SDL_GetError());
        return 1;
    }
    

    
    font = TTF_OpenFont("sf-pro-text-regular.ttf", FONT_SIZE);
    if (font == nullptr) {
        printf("Could not load font: %s\n", TTF_GetError());
        return 1;
    }

    font_small = TTF_OpenFont("sf-pro-text-regular.ttf", FONT_SIZE_SMALL);
    if (font == nullptr) {
        printf("Could not load font: %s\n", TTF_GetError());
        return 1;
    }

    

    SDL_Rect srcRect = { 0, 0, IMG_WIDTH, IMG_HEIGHT };
    SDL_Rect destRect = { 0, 0, IMG_WIDTH * SCALE_FACTOR, IMG_HEIGHT * SCALE_FACTOR };
//     SDL_Rect textRect = { TEXT_POS_X, TEXT_POS_Y, text_surface->w, text_surface->h };

    SDL_Event event;
    int running = 1;
    int counter = 0;
    char str[ENOUGH];
    char str2[ENOUGH];
    char str3[ENOUGH];
    float correct = 0;
    float accuracy = 0;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }
        }

        SDL_Surface *text_surface = nullptr;
        SDL_Texture *text_texture = nullptr;

        SDL_Surface *text_surface2 = nullptr;
        SDL_Texture *text_texture2 = nullptr;
        SDL_Color textcol2;

        SDL_Surface *text_surface3 = nullptr;
        SDL_Texture *text_texture3 = nullptr;

        SDL_Surface *text_surface4 = nullptr;
        SDL_Texture *text_texture4 = nullptr;

        

        std::pair<uint, Eigen::VectorXf> result = nn.predict(x_test.at(counter));
        uint predicted;
        Eigen::VectorXf probabilities;
        std::tie(predicted, probabilities) = result;
        uint actual = one_hot_decode(y_test.at(counter));
        if (actual == predicted) {
            correct++;
            textcol2 = textGreen;
        } else textcol2 = textRed;

        accuracy = correct/(float)counter;

        // "Label: %hhu\nPredicted: %hhu\n", + some correct or incorrect text and running correct ratio and percentage

        // printf("%d", actual);
        sprintf(str, "Label: %hhu", (uint8_t)actual);
        sprintf(str2, "Predicted: %hhu", (uint8_t)predicted);
        sprintf(str3, "Accuracy: %f\n%d/%d", accuracy, (int)correct, counter);


        text_surface = TTF_RenderText_Blended(font, str, textColor);
        if (text_surface == nullptr) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
        if (text_texture == nullptr) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface2 = TTF_RenderText_Blended(font, str2, textcol2);
        if (text_surface2 == nullptr) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture2 = SDL_CreateTextureFromSurface(renderer, text_surface2);
        if (text_texture2 == nullptr) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface3 = TTF_RenderText_Blended_Wrapped(font, str3, textColor, 0);
        if (text_surface3 == nullptr) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture3 = SDL_CreateTextureFromSurface(renderer, text_surface3);
        if (text_texture3 == nullptr) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }

        text_surface4 = TTF_RenderText_Blended_Wrapped(font_small, nn.nnInfoString().c_str(), textColor, 0);
        if (text_surface4 == nullptr) {
            printf("Could not create text surface: %s\n", TTF_GetError());
            return 1;
        }

        text_texture4 = SDL_CreateTextureFromSurface(renderer, text_surface4);
        if (text_texture4 == nullptr) {
            printf("Could not create text texture: %s\n", SDL_GetError());
            return 1;
        }


        texture = createGrayscaleImageTexture(x_test.at(counter).data(), renderer);
        if (texture == nullptr) {
            return 1;
        }



        // We don't need the surface anymore
        SDL_FreeSurface(text_surface);
        SDL_Rect textRect = { TEXT_POS_X, TEXT_POS_Y, text_surface->w, text_surface->h };
        SDL_Rect textRect2 = { TEXT_POS_X, TEXT_POS_Y+40, text_surface2->w, text_surface2->h };
        SDL_Rect textRect3 = { TEXT_POS_X, TEXT_POS_Y+120, text_surface3->w, text_surface3->h };
        SDL_Rect textRect4 = { TEXT_POS_X, TEXT_POS_Y-65, text_surface4->w, text_surface4->h };



        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_RenderCopy(renderer, texture, &srcRect, &destRect);
        SDL_RenderCopy(renderer, text_texture, nullptr, &textRect);
        SDL_RenderCopy(renderer, text_texture2, nullptr, &textRect2);
        SDL_RenderCopy(renderer, text_texture3, nullptr, &textRect3);
        SDL_RenderCopy(renderer, text_texture4, nullptr, &textRect4);
        

        SDL_RenderPresent(renderer);

        if(++counter > y_test.size()) break;
        std::this_thread::sleep_for(0.1s);
    }

    // free()
    // Clean up
    TTF_CloseFont(font);
    SDL_DestroyTexture(texture);
    // SDL_DestroyTexture(text_texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}

