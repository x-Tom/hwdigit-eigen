#pragma once

#include <random>
#include <fstream>
#include <algorithm>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>


constexpr unsigned IMAGE_WIDTH  = 28;
constexpr unsigned IMAGE_HEIGHT = 28;
// using pixel_vector = Eigen::Vector<float, IMAGE_WIDTH*IMAGE_HEIGHT>;
using pixel_vector = Eigen::VectorXf;
// using label_vector = Eigen::Vector<float, 10>;
using label_vector = Eigen::VectorXf;
// Activation functions and their derivatives

#define ENOUGH 50
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define SCALE_FACTOR 10
#define WINDOW_WIDTH 560
#define WINDOW_HEIGHT 280
#define TEXT_POS_X 300
// #define TEXT_POS_Y 50
#define TEXT_POS_Y 70


SDL_Texture* createGrayscaleImageTexture(float* array, SDL_Renderer *renderer) {

    SDL_Surface* surface = SDL_CreateRGBSurface(0, IMG_WIDTH, IMG_HEIGHT, 32, 0, 0, 0, 0);
    if (surface == NULL) {
        printf("Could not create surface: %s\n", SDL_GetError());
        return NULL;
    }

    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            Uint8 gray = (Uint8)(255*array[y * IMG_WIDTH + x]);
            Uint32 color = SDL_MapRGB(surface->format, gray, gray, gray);
            ((Uint32*)surface->pixels)[y * IMG_WIDTH + x] = color;
        }
    }

    // Create a texture from the surface
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) {
        printf("Could not create texture: %s\n", SDL_GetError());
        return NULL;
    }

    // We don't need the surface anymore
    SDL_FreeSurface(surface);

    // Return the texture
    return texture;
}


std::vector<pixel_vector> read_mnist_pixel_data(const std::string& filename, uint n){
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("File not opening!");
    
    std::vector<pixel_vector> pixel_arr(n); 
    for(int k = 0; k < n; k++){
        pixel_arr[k] = Eigen::VectorXf::Zero(IMAGE_WIDTH*IMAGE_HEIGHT);
        for(int i = 0; i < IMAGE_HEIGHT; i++){
            for(int j = 0; j < IMAGE_WIDTH; j++){
                float f;
                (file >> f) ? pixel_arr[k](i*IMAGE_WIDTH+j) = f : pixel_arr[k](i*IMAGE_WIDTH+j) = 0;
            }
        }
        pixel_arr[k] /= 255;
        // pixel_arr[k] = (pixel_arr[k].array() - pixel_arr[k].mean() )/stddev_vec(pixel_arr[k]);
    }

    file.close();
    return pixel_arr;
}

label_vector one_hot_encoding(int n){
    label_vector vec = label_vector::Zero(10);
    if(n == -1) return vec;
    vec(n) = 1;
    return vec;
}

uint one_hot_decode(const label_vector& vec){
    Eigen::VectorXf::Index idx;
    vec.maxCoeff(&idx);
    return static_cast<uint>(idx);
}

std::vector<label_vector> read_mnist_label_data(const std::string& filename, uint n){
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("File not opening!");
    
    std::vector<label_vector> label_arr(n); 
    for(int k = 0; k < n; k++){
        int digit;
        label_arr[k] = (file >> digit) ? one_hot_encoding(digit) : one_hot_encoding(-1);
    }
    file.close();
    return label_arr;
}

template <typename T1, typename T2>
void two_vec_shuffle(std::vector<T1>& arr1, std::vector<T2>& arr2, std::mt19937& rng){
    if(arr1.size() != arr2.size()) {
        throw std::invalid_argument("Array sizes are not equal!");
    }
    for(uint i = 0; i < arr1.size(); i++){
        std::uniform_int_distribution<> dis(i, arr1.size() - 1);
        uint index = dis(rng);

        std::swap(arr1.at(i), arr1.at(index));
        std::swap(arr2.at(i), arr2.at(index));
    }
}
