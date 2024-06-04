#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"
#define EPSILON 1e-5 // A small number for perturbing the weights

// Function declarations
void update_parameters(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);
void check_gradients();  // Declare the gradient checking function

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

// Parameters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double initial_learning_rate;
double final_learning_rate;
double learning_rate; // Current learning rate

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}


// void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
//     batch_size = cmd_line_batch_size;
//     learning_rate = cmd_line_learning_rate;
//     total_epochs = cmd_line_total_epochs;
    
//     num_batches = total_epochs * (N_TRAINING_SET / batch_size);
//     printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
//            total_epochs, batch_size, num_batches, learning_rate);
// }

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    initial_learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tinitial_learning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, initial_learning_rate);
}

// void run_optimisation(void){
//     unsigned int training_sample = 0;
//     unsigned int total_iter = 0;
//     double obj_func = 0.0;
//     unsigned int epoch_counter = 0;
//     double test_accuracy = 0.0;  //evaluate_testing_accuracy();
//     double mean_loss = 0.0;
    
//     // Run optimiser - update parameters after each minibatch
//     for (int i=0; i < num_batches; i++){
//         for (int j = 0; j < batch_size; j++){

//             // Evaluate accuracy on testing set (expensive, evaluate infrequently)
//             if (total_iter % log_freq == 0 || total_iter == 0){
//                 if (total_iter > 0){
//                     mean_loss = mean_loss/((double) log_freq);
//                 }
                
//                 test_accuracy = evaluate_testing_accuracy();
//                 print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);

//                 // Reset mean_loss for next reporting period
//                 mean_loss = 0.0;
//             }
            
//             // Evaluate forward pass and calculate gradients
//             obj_func = evaluate_objective_function(training_sample);
//             mean_loss+=obj_func;

//             // Update iteration counters (reset at end of training set to allow multiple epochs)
//             total_iter++;
//             training_sample++;
//             // On epoch completion:
//             if (training_sample == N_TRAINING_SET){
//                 training_sample = 0;
//                 epoch_counter++;
//             }
//         }
        
//         // Update weights on batch completion
//         update_parameters(batch_size);
//     }
    
//     // Print final performance
//     test_accuracy = evaluate_testing_accuracy();
//     print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);
// }

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;
    double mean_loss = 0.0;
    double decay_factor = 0.1;  // Adjust this to control how quickly the learning rate decays
    
    // Run optimiser - update parameters after each minibatch
    for (int i = 0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){

            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss / ((double) log_freq);
                }
                
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);
                mean_loss = 0.0;
            }
            
            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;

            total_iter++;
            training_sample++;
            
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
                learning_rate = initial_learning_rate * pow((1.0 - decay_factor), epoch_counter);  // Exponential decay
                printf("Updated learning rate: %f\n", learning_rate);
            }
        }
        
        update_parameters(batch_size);
    }
    
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double) log_freq)), test_accuracy);
}

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

void compute_and_compare_gradients(weight_struct_t weights[][N_NEURONS_L1], unsigned int n_pre, unsigned int n_post, unsigned int sample) {
    double original_weight, perturbed_loss_plus, perturbed_loss_minus, original_loss, dw_numerical;

    original_loss = evaluate_objective_function(sample);

    for (int i = 0; i < n_pre; i++) {
        for (int j = 0; j < n_post; j++) {
            original_weight = weights[i][j].w;

            // Perturb weight positively
            weights[i][j].w = original_weight + EPSILON;
            perturbed_loss_plus = evaluate_objective_function(sample);

            // Perturb weight negatively
            weights[i][j].w = original_weight - EPSILON;
            perturbed_loss_minus = evaluate_objective_function(sample);

            // Compute numerical gradient
            dw_numerical = (perturbed_loss_plus - perturbed_loss_minus) / (2 * EPSILON);

            // Restore the original weight
            weights[i][j].w = original_weight;

            // Optionally log or compare here
            printf("Analytical Gradient: %f, Numerical Gradient: %f\n", weights[i][j].dw, dw_numerical);
        }
    }
}


// void update_parameters(unsigned int batch_size){
//     double alpha = 0.2;  // Momentum coefficient

//     // Iterate over weight matrices between each pair of connected layers and apply momentum
//     for (int i = 0; i < N_NEURONS_LI; i++) {
//         for (int j = 0; j < N_NEURONS_L1; j++) {
//             w_LI_L1[i][j].v = alpha * w_LI_L1[i][j].v + (learning_rate * w_LI_L1[i][j].dw / batch_size);
//             w_LI_L1[i][j].w -= w_LI_L1[i][j].v;
//             w_LI_L1[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L1; i++) {
//         for (int j = 0; j < N_NEURONS_L2; j++) {
//             w_L1_L2[i][j].v = alpha * w_L1_L2[i][j].v + (learning_rate * w_L1_L2[i][j].dw / batch_size);
//             w_L1_L2[i][j].w -= w_L1_L2[i][j].v;
//             w_L1_L2[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L2; i++) {
//         for (int j = 0; j < N_NEURONS_L3; j++) {
//             w_L2_L3[i][j].v = alpha * w_L2_L3[i][j].v + (learning_rate * w_L2_L3[i][j].dw / batch_size);
//             w_L2_L3[i][j].w -= w_L2_L3[i][j].v;
//             w_L2_L3[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L3; i++) {
//         for (int j = 0; j < N_NEURONS_LO; j++) {
//             w_L3_LO[i][j].v = alpha * w_L3_LO[i][j].v + (learning_rate * w_L3_LO[i][j].dw / batch_size);
//             w_L3_LO[i][j].w -= w_L3_LO[i][j].v;
//             w_L3_LO[i][j].dw = 0; // Reset gradient after update
//         }
//     }
// }

// void update_parameters(unsigned int batch_size){
//     double epsilon = 1e-8;  // Small number to prevent division by zero

//     for (int i = 0; i < N_NEURONS_LI; i++) {
//         for (int j = 0; j < N_NEURONS_L1; j++) {
//             // printf("Before Update - Weight w_LI_L1[%d][%d]: %f, Gradient: %f\n", i, j, w_LI_L1[i][j].w, w_LI_L1[i][j].dw);
//             w_LI_L1[i][j].g2 += pow(w_LI_L1[i][j].dw, 2);
//             w_LI_L1[i][j].w -= (learning_rate / (sqrt(w_LI_L1[i][j].g2) + epsilon)) * w_LI_L1[i][j].dw;
//             // printf("After Update - Adjusted LR: %f, New Weight w_LI_L1[%d][%d]: %f\n", initial_learning_rate / (sqrt(w_LI_L1[i][j].g2) + 1e-8), i, j, w_LI_L1[i][j].w);
//             w_LI_L1[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L1; i++) {
//         for (int j = 0; j < N_NEURONS_L2; j++) {
//             // printf("Before Update - Weight w_L1_L2[%d][%d]: %f, Gradient: %f\n", i, j, w_L1_L2[i][j].w, w_L1_L2[i][j].dw);
//             w_L1_L2[i][j].g2 += pow(w_L1_L2[i][j].dw, 2);
//             w_L1_L2[i][j].w -= (learning_rate / (sqrt(w_L1_L2[i][j].g2) + epsilon)) * w_L1_L2[i][j].dw;
//             // printf("After Update - Adjusted LR: %f, New Weight w_L1_L2[%d][%d]: %f\n", initial_learning_rate / (sqrt(w_L1_L2[i][j].g2) + 1e-8), i, j, w_L1_L2[i][j].w);
//             w_L1_L2[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L2; i++) {
//         for (int j = 0; j < N_NEURONS_L3; j++) {
//             // printf("Before Update - Weight w_L2_L3[%d][%d]: %f, Gradient: %f\n", i, j, w_L2_L3[i][j].w, w_L2_L3[i][j].dw);
//             w_L2_L3[i][j].g2 += pow(w_L2_L3[i][j].dw, 2);
//             w_L2_L3[i][j].w -= (learning_rate / (sqrt(w_L2_L3[i][j].g2) + epsilon)) * w_L2_L3[i][j].dw;
//             // printf("After Update - Adjusted LR: %f, New Weight w_L2_L3[%d][%d]: %f\n", initial_learning_rate / (sqrt(w_L2_L3[i][j].g2) + 1e-8), i, j, w_L2_L3[i][j].w);
//             w_L2_L3[i][j].dw = 0; // Reset gradient after update
//         }
//     }

//     for (int i = 0; i < N_NEURONS_L3; i++) {
//         for (int j = 0; j < N_NEURONS_LO; j++) {
//             // printf("Before Update - Weight w_L3_LO[%d][%d]: %f, Gradient: %f\n", i, j, w_L3_LO[i][j].w, w_L3_LO[i][j].dw);
//             w_L3_LO[i][j].g2 += pow(w_L3_LO[i][j].dw, 2);
//             w_L3_LO[i][j].w -= (learning_rate / (sqrt(w_L3_LO[i][j].g2) + epsilon)) * w_L3_LO[i][j].dw;
//             // printf("After Update - Adjusted LR: %f, New Weight w_L3_LO[%d][%d]: %f\n", initial_learning_rate / (sqrt(w_L3_LO[i][j].g2) + 1e-8), i, j, w_L3_LO[i][j].w);
//             w_L3_LO[i][j].dw = 0; // Reset gradient after update
//         }
//     }
// }

void update_parameters(unsigned int batch_size) {
    static double G_LI_L1[N_NEURONS_LI][N_NEURONS_L1] = {0}; // Accumulate squared gradients
    static double G_L1_L2[N_NEURONS_L1][N_NEURONS_L2] = {0};
    static double G_L2_L3[N_NEURONS_L2][N_NEURONS_L3] = {0};
    static double G_L3_LO[N_NEURONS_L3][N_NEURONS_LO] = {0};

    for (int i = 0; i < N_NEURONS_LI; i++) {
        for (int j = 0; j < N_NEURONS_L1; j++) {
            // Update accumulated gradient squares
            G_LI_L1[i][j] += pow(w_LI_L1[i][j].dw / batch_size, 2);

            // Compute adjusted learning rate
            double adjusted_lr = learning_rate / sqrt(G_LI_L1[i][j] + EPSILON);

            // Update weights
            w_LI_L1[i][j].w -= adjusted_lr * w_LI_L1[i][j].dw / batch_size;

            // Reset gradient after update
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++) {
        for (int j = 0; j < N_NEURONS_L2; j++) {
            G_L1_L2[i][j] += pow(w_L1_L2[i][j].dw / batch_size, 2);
            double adjusted_lr = learning_rate / sqrt(G_L1_L2[i][j] + EPSILON);
            w_L1_L2[i][j].w -= adjusted_lr * w_L1_L2[i][j].dw / batch_size;
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++) {
        for (int j = 0; j < N_NEURONS_L3; j++) {
            G_L2_L3[i][j] += pow(w_L2_L3[i][j].dw / batch_size, 2);
            double adjusted_lr = learning_rate / sqrt(G_L2_L3[i][j] + EPSILON);
            w_L2_L3[i][j].w -= adjusted_lr * w_L2_L3[i][j].dw / batch_size;
            w_L2_L3[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            G_L3_LO[i][j] += pow(w_L3_LO[i][j].dw / batch_size, 2);
            double adjusted_lr = learning_rate / sqrt(G_L3_LO[i][j] + EPSILON);
            w_L3_LO[i][j].w -= adjusted_lr * w_L3_LO[i][j].dw / batch_size;
            w_L3_LO[i][j].dw = 0;
        }
    }
}







