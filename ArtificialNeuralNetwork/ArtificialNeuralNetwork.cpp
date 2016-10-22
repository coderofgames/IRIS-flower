// ArtificialNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Timer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "NeuralNetworks.h"









template< class NeuralNetwork >
void Compute_IRIS_data_version_2_(int num_iterations, CSV &iris_data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
{
	Timer timer;

	int input_data_size = 1;
	int num_inputs = 4;
	int num_hidden = 8;
	int num_hidden_2 = 3;
	int num_outputs = 1;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	float output_error = 0.0f;



	float alpha = 0.3f;
	float beta = 0.05f;

	float sum_squared_errors = 0.0f;

	vector<int> layer_sizes;
	layer_sizes.push_back(num_inputs);
	layer_sizes.push_back(num_hidden);
	layer_sizes.push_back(num_hidden);
	layer_sizes.push_back(num_hidden_2);

	NeuralNetwork *neuralNet = new NeuralNetwork(layer_sizes);

	cout << endl;
	cout << "Training, please wait ..." << endl;

	timer.Start();

	matrix Cost_Matrix(1, 3);

	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = iris_data.iris_data[training_set[q]][0];
			input_matrix(0, 1) = iris_data.iris_data[training_set[q]][1];
			input_matrix(0, 2) = iris_data.iris_data[training_set[q]][2];
			input_matrix(0, 3) = iris_data.iris_data[training_set[q]][3];

			// formulate the correct solution vector
			matrix expected(1, 3);
			for (int p = 0; p < 3; p++)
			{
				if ((int)iris_data.iris_data[training_set[q]][4] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}
			// feed forward
			matrix output = neuralNet->FeedForward(input_matrix);
			
			matrix errors = expected - output;
			
			matrix errors2 = errors;

			Cost_Matrix = Cost_Matrix + pow(errors2, 2);

			//errors = ln(output) * (-1);

			for (int p = 0; p < errors.NumCols(); p++)
			{
				//if (abs(errors(0, p)) > 0.15)
				{
					//errors(0, p) = -log(output(0, p));// errors(0, p) * sigmoid_deriv(output(0, p));
					errors(0, p) = errors(0, p) * sigmoid_deriv(output(0, p));
				}
				//else errors(0, p) = 0.0f;
			}

			// back propagate output
			neuralNet->BackPropagateErrors(errors);

			// weight deltas
			neuralNet->ComputeDeltas(alpha, beta);

			// update weights
			neuralNet->UpdateWeights();

		}
		Cost_Matrix = Cost_Matrix / training_set.size();

		if ((Cost_Matrix(0, 0) < 0.01) && (Cost_Matrix(0, 1) < 0.01) && (Cost_Matrix(0, 2) < 0.01))
		{
			cout << "finished training because sum of squared errors is low at iteration p: " << mm << endl;
			break;
		}

		Cost_Matrix.ToZero();
		

	}
	timer.Update();
	timer.Stop();
	double time_taken = timer.GetTimeDelta();
		cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	for (int q = 0; q < test_set.size(); q++)
	{
		input_matrix(0, 0) = iris_data.iris_data[test_set[q]][0];
		input_matrix(0, 1) = iris_data.iris_data[test_set[q]][1];
		input_matrix(0, 2) = iris_data.iris_data[test_set[q]][2];
		input_matrix(0, 3) = iris_data.iris_data[test_set[q]][3];

		matrix test_output = neuralNet->FeedForward(input_matrix);

		if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = (int)iris_data.iris_data[test_set[q]][4];

		int found_type = 0;

		if ((test_output(0, 0) > 0.8) && (test_output(0, 1) < 0.2) && (test_output(0, 2) < 0.2))
		{
			found_type = 0;
		}
		if ((test_output(0, 0) < 0.2) && (test_output(0, 1) > 0.8) && (test_output(0, 2) < 0.2))
		{
			found_type = 1;
		}
		if ((test_output(0, 0) < 0.2) && (test_output(0, 1) < 0.2) && (test_output(0, 2)> 0.8))
		{
			found_type = 2;
		}

		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}




template< class Layer >
void Compute_IRIS_data_version_0_(int num_iterations,  CSV &iris_data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes)
{
	Timer timer;



	int input_data_size = 1;
	int num_inputs = 4;
	int num_hidden = 8;
	int num_hidden_2 = 3;
	int num_outputs = 1;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	float output_error = 0.0f;



	float alpha = 0.5f;
	float beta = 0.05f;

	float sum_squared_errors = 0.0f;

	Layer hidden(num_hidden, num_inputs);
	hidden.init_random_sample_weights_iris();

	Layer hidden_2(num_hidden_2, num_hidden);
	hidden_2.init_random_sample_weights_iris();

	Layer output(num_outputs, num_hidden_2);
	output.init_random_sample_weights_iris();


	

	//	indexes
	//vector< matrix > output_buffer;

	cout << endl;
	cout << "Training, please wait ..." << endl;
	//return;
	timer.Start();

	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;



	for (int mm = 0; mm< num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = iris_data.iris_data[training_set[q]][0];
			input_matrix(0, 1) = iris_data.iris_data[training_set[q]][1];
			input_matrix(0, 2) = iris_data.iris_data[training_set[q]][2];
			input_matrix(0, 3) = iris_data.iris_data[training_set[q]][3];

			matrix expected(1, 3);
			for (int p = 0; p < 3; p++)
			{
				if ((int)iris_data.iris_data[training_set[q]][4] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}




			// feed forward

			hidden.FeedForward(input_matrix);
			hidden_2.FeedForward(hidden.neurons_);

			// compute the cost function for this training sample
			matrix errors = expected - hidden_2.neurons_;

			for (int p = 0; p < errors.NumCols(); p++)
			{
				if (abs(errors(0, p)) > 0.2)
				{
					errors(0, p) = errors(0, p) * sigmoid_deriv(hidden_2.neurons_(0, p));
				}
				else errors(0, p) = 0.0f;
			}
			

			// back propagate output

			hidden_2.BackPropogate_output(errors);
			hidden.BackPropogate(hidden_2.weights_, hidden_2.deltas_);

			// weight deltas

			hidden_2.ComputeWeightDeltas(hidden.neurons_, alpha, beta);
			hidden.ComputeWeightDeltas(input_matrix, alpha, beta);


			// update weights

			hidden_2.UpdateWeights();
			hidden.UpdateWeights();


		}


	}
	timer.Update();
	timer.Stop();
	cout << "Finished training, Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;


	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	for (int q = 0; q < test_set.size(); q++)
	{
		/*input_matrix(0, 0) = iris_data.iris_data[q][0];
		input_matrix(0, 1) = iris_data.iris_data[q][1];
		input_matrix(0, 2) = iris_data.iris_data[q][2];
		input_matrix(0, 3) = iris_data.iris_data[q][3];*/

		input_matrix(0, 0) = iris_data.iris_data[test_set[q]][0];
		input_matrix(0, 1) = iris_data.iris_data[test_set[q]][1];
		input_matrix(0, 2) = iris_data.iris_data[test_set[q]][2];
		input_matrix(0, 3) = iris_data.iris_data[test_set[q]][3];
		//input_matrix(0, 2) = -1.0f; // bias is always -1



		hidden.FeedForward(input_matrix);
		hidden_2.FeedForward(hidden.neurons_);

		//int actual_type = (int)iris_data.iris_data[q][4];
		int actual_type = (int)iris_data.iris_data[test_set[q]][4];

		int found_type = 0;

		if ((hidden_2.neurons_(0, 0) > 0.8) && (hidden_2.neurons_(0, 1) < 0.2) && (hidden_2.neurons_(0, 2) < 0.2))
		{
			found_type = 0;
		}
		if ((hidden_2.neurons_(0, 0) < 0.2) && (hidden_2.neurons_(0, 1) > 0.8) && (hidden_2.neurons_(0, 2) < 0.2))
		{
			found_type = 1;
		}
		if ((hidden_2.neurons_(0, 0) < 0.2) && (hidden_2.neurons_(0, 1) < 0.2) && (hidden_2.neurons_(0, 2)> 0.8))
		{
			found_type = 2;
		}

	//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}

//#define VERBOSE
template< class Layer >
void Compute_IRIS_data_version_1_(int num_iterations, CSV &iris_data ,vector<int> &training_set, vector<int> &test_set, vector<int> &indexes)
{
	Timer timer;


	int input_data_size = 1;
	int num_inputs = 4;
	int num_hidden = 6;
	int num_hidden_2 = 6;
	int num_outputs = 3;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	float output_error = 0.0f;



	float alpha = 0.5f;
	float beta = 0.05f;

	float sum_squared_errors = 0.0f;

	Layer hidden(num_hidden, num_inputs);
	hidden.init_random_sample_weights_iris();

	Layer hidden_2(num_hidden_2, num_hidden);
	hidden_2.init_random_sample_weights_iris();

	Layer output(num_outputs, num_hidden_2);
	output.init_random_sample_weights_iris();



	cout << endl;
	cout << "Training, please wait ..." << endl;
	//return;
	timer.Start();

	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;

	

	for (int mm = 0; mm < num_iterations; mm++)
	{
		
		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = iris_data.iris_data[ training_set[q] ][0];
			input_matrix(0, 1) = iris_data.iris_data[ training_set[q] ][1];
			input_matrix(0, 2) = iris_data.iris_data[ training_set[q] ][2];
			input_matrix(0, 3) = iris_data.iris_data[ training_set[q] ][3];
			
			// feed forward
			hidden.FeedForward(input_matrix);
			hidden_2.FeedForward(hidden.neurons_);
			output.FeedForward(hidden_2.neurons_);

			matrix expected(1, 3);
			for (int p = 0; p < 3; p++)
			{
				if ((int)iris_data.iris_data[training_set[q]][4] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}

			// compute the output error 

			matrix errors = expected - output.neurons_;

			for (int p = 0; p < errors.NumCols(); p++)
			{
				if (abs(errors(0, p)) > 0.2)
				{
					errors(0, p) = errors(0, p) * sigmoid_deriv(output.neurons_(0, p));
				}
				else errors(0, p) = 0.0;
			}

			// back propagate errors
			
			output.BackPropogate_output( errors);
			hidden_2.BackPropogate(output.weights_, output.deltas_);
			hidden.BackPropogate(hidden_2.weights_, hidden_2.deltas_);


			// weight deltas

			output.ComputeWeightDeltas(hidden_2.neurons_, alpha, beta);
			hidden_2.ComputeWeightDeltas(hidden.neurons_, alpha, beta);
			hidden.ComputeWeightDeltas(input_matrix, alpha, beta);


			// update weights

			output.UpdateWeights();
			hidden_2.UpdateWeights();
			hidden.UpdateWeights();

		}


	}
	timer.Update();
	timer.Stop();
	cout << "Finished training, Total calculation performed in " << timer.GetTimeDelta() << " seconds" << endl;


	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	for (int q = 0; q < test_set.size(); q++)
	{		
		input_matrix(0, 0) = iris_data.iris_data[test_set[q]][0];
		input_matrix(0, 1) = iris_data.iris_data[test_set[q]][1];
		input_matrix(0, 2) = iris_data.iris_data[test_set[q]][2];
		input_matrix(0, 3) = iris_data.iris_data[test_set[q]][3];
		//input_matrix(0, 2) = -1.0f; // bias is always -1

		hidden.FeedForward(input_matrix);
		hidden_2.FeedForward(hidden.neurons_);
		output.FeedForward(hidden_2.neurons_);

		int actual_type = (int)iris_data.iris_data[test_set[q]][4];

		int found_type = 0;

			if ((output.neurons_(0, 0) > 0.8) && (output.neurons_(0, 1) < 0.2) && (output.neurons_(0, 2) < 0.2))
			{
				found_type = 0;
			}
			if ((output.neurons_(0, 0) < 0.2) && (output.neurons_(0, 1) > 0.8) && (output.neurons_(0, 2) < 0.2))
			{
				found_type = 1;
			}
			if ((output.neurons_(0, 0) < 0.2) && (output.neurons_(0, 1) < 0.2) && (output.neurons_(0, 2)> 0.8))
			{
				found_type = 2;
			}

//			cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

			if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of "<< test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}


void Compute_IRIS_data_version_3_(int num_iterations, vector<vector<float>> &data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
{
	Timer timer;

	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 9;
	int num_hidden2 = 3;
	int num_hidden3 = 8;
	int num_outputs = 3;

	// ==========================================
	matrix input_matrix(1, num_inputs);

	float alpha = 0.5f;
	float beta = 0.2;// 5f; // this does not appear to be working on either of these problems

	float sum_squared_errors = 0.0f;

	vector<int> layer_sizes;
	layer_sizes.push_back(num_inputs);
	layer_sizes.push_back(num_hidden);
	//layer_sizes.push_back(num_hidden2);
	//layer_sizes.push_back(num_hidden3);
	layer_sizes.push_back(num_outputs);

	Vanilla_Layer::Linked_Layer::NeuralNetwork *neuralNet = 
		new Vanilla_Layer::Linked_Layer::NeuralNetwork(layer_sizes);

	//neuralNet->input_layer[3]->alpha = 0.1;
	neuralNet->input_layer[2]->alpha = 0.3;
	neuralNet->input_layer[1]->alpha = 0.5;
	neuralNet->input_layer[0]->alpha = 0.7;

	cout << endl;
	cout << "Training, please wait ..." << endl;

	timer.Start();

	float tolerance = 0.15; // the new hyperparameter

	matrix Cost_Matrix(1, 3);
	matrix expected(1, 3);
	matrix output(1,3);
	
	matrix errors2(1, 3);
	matrix errors(1, 3);
	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = data[ training_set[q] ][0];
			input_matrix(0, 1) = data[ training_set[q] ][1];

			// formulate the correct output vector
			
			for (int p = 0; p < 3; p++)
			{
				if ((int)data[training_set[q]][2] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}

			// feed forward

			output = neuralNet->FeedForward(input_matrix);

			errors = expected - output;
			
			errors2 = errors;

			Cost_Matrix = Cost_Matrix + pow(errors2, 2);

			for (int p = 0; p < errors.NumCols(); p++)
			{
				if (abs(errors(0, p)) > tolerance)
				{
					errors(0, p) = errors(0, p) * sigmoid_deriv(output(0, p));
				}
				else errors(0, p) = 0.0f;
			}

			// back propagate output
			neuralNet->BackPropagateErrors(errors);

			// weight deltas
			neuralNet->ComputeDeltas(alpha, beta);

			// update weights
			neuralNet->UpdateWeights();

		}

		Cost_Matrix = Cost_Matrix / training_set.size();

		if ((Cost_Matrix(0, 0) < 0.001) && (Cost_Matrix(0, 1) < 0.001) && (Cost_Matrix(0, 2) < 0.001))
		{
			cout << "finished training because sum of squared errors is low at iteration p: " << mm << endl;
			break;
		}

		Cost_Matrix.ToZero();
	}

	timer.Update();
	timer.Stop();
	
	double time_taken = timer.GetTimeDelta();
	cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	tolerance += 0.2;

	matrix test_output(1, 3);
	for (int q = 0; q < test_set.size(); q++)
	{
		input_matrix(0, 0) = data[test_set[q]][0];
		input_matrix(0, 1) = data[test_set[q]][1];

		// test the neural network

		test_output = neuralNet->FeedForward(input_matrix);

		if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = (int)data[test_set[q]][2];

		int found_type = 0;

		if ((test_output(0, 0) > (1 - tolerance)) && (test_output(0, 1) < tolerance) && (test_output(0, 2) < tolerance))
		{
			found_type = 0;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) > (1 - tolerance)) && (test_output(0, 2) < tolerance))
		{
			found_type = 1;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) < tolerance) && (test_output(0, 2)> (1 - tolerance)))
		{
			found_type = 2;
		}

		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}
void Compute_IRIS_data_version_5_(int num_iterations, vector<vector<float>> &data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
{
	Timer timer;



	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 20;
	int num_hidden2 = 18;
	int num_hidden3 = 8;
	int num_outputs = 3;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	float output_error = 0.0f;



	float alpha = 0.5f;
	float beta = 0.05f;

	float reg = 0.001;

	float sum_squared_errors = 0.0f;

	vector<int> layer_sizes;
	layer_sizes.push_back(num_inputs);
	layer_sizes.push_back(num_hidden);
	//layer_sizes.push_back(num_hidden2);
	//layer_sizes.push_back(num_hidden3);
	layer_sizes.push_back(num_outputs);

	Improved_Layer::Linked_Layer_Loop_Eval::NeuralNetwork *neuralNet =
		new Improved_Layer::Linked_Layer_Loop_Eval::NeuralNetwork(layer_sizes);








	//	indexes
	//vector< matrix > output_buffer;

	cout << endl;
	cout << "Training, please wait ..." << endl;
	//return;
	timer.Start();

	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;

	float tolerance = 0.05;

	matrix Cost_Matrix(1, 3);

	matrix scores(training_set.size(), 3);
	matrix expect(training_set.size(), 3);
	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = data[training_set[q]][0];
			input_matrix(0, 1) = data[training_set[q]][1];


			matrix expected(1, 3);
			for (int p = 0; p < 3; p++)
			{
				if ((int)data[training_set[q]][2] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}


			// feed forward


			matrix output = neuralNet->FeedForward_ReLU(input_matrix);


			float sum_outputs = 0.0;
			for (int p = 0; p < 3; p++)
			{
				scores(q, p) = output(0, p);

				expect(q, p) = expected(0, p);
			}





			// back propagate output
			matrix probs_(1, 3);


			for (int i = 0; i < 3; i++)
			{
				probs_(0, i) = output(0, i);

				if (expected(0, i) == 1.0)
					probs_(0, i) = probs_(0, i) - 1;


				//if (expected(0, i) == 1.0)
				//{
				//	probs_(0, i) = probs(0, i) - 1;
				//}
			}
			neuralNet->BackPropagateErrors(probs_);


			// weight deltas
			neuralNet->ComputeDeltas(probs_, expected, training_set.size(), alpha, beta);

			// update weights
			neuralNet->UpdateWeights(reg);




		}

		//	scores = neuralNet->Compute_Probabilities(scores);
		matrix probs = scores;
		//cout << "scores: " << scores.NumRows() << ", " << scores.NumCols() << endl;

		scores = neuralNet->Compute_Log_Probabilities(probs, expect);


		float data_loss = neuralNet->Compute_Data_Loss(scores, training_set.size());

		float reg_loss = neuralNet->Compute_Regularization_Loss(reg);

		float total_loss = data_loss + reg_loss;

		if (mm % 20 == 0)
			cout << "total loss: " << total_loss << ", data_loss: " << data_loss << ", reg loss: " << reg_loss << endl;

		for (int q = 0; q < training_set.size(); q++)
		{

		}







	}
	timer.Update();
	timer.Stop();
	double time_taken = timer.GetTimeDelta();
	cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	for (int q = 0; q < test_set.size(); q++)
	{


		input_matrix(0, 0) = data[test_set[q]][0];
		input_matrix(0, 1) = data[test_set[q]][1];

		//input_matrix(0, 2) = -1.0f; // bias is always -1



		matrix test_output = neuralNet->FeedForward_ReLU(input_matrix);

		if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = (int)data[test_set[q]][2];

		int found_type = 0;

		if ((test_output(0, 0) > (1 - tolerance)) && (test_output(0, 1) < tolerance) && (test_output(0, 2) < tolerance))
		{
			found_type = 0;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) > (1 - tolerance)) && (test_output(0, 2) < tolerance))
		{
			found_type = 1;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) < tolerance) && (test_output(0, 2) > (1 - tolerance)))
		{
			found_type = 2;
		}

		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}


void Compute_IRIS_data_version_4_(int num_iterations, vector<vector<float>> &data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes, double &time_)
{
	Timer timer;



	int input_data_size = 1;
	int num_inputs = 2;
	int num_hidden = 20;
	int num_hidden2 = 18;
	int num_hidden3 = 8;
	int num_outputs = 3;

	// ==========================================
	matrix input_matrix(1, num_inputs);


	float output_error = 0.0f;



	float alpha = 0.5f;
	float beta = 0.05f;

	float reg = 0.001;

	float sum_squared_errors = 0.0f;

	vector<int> layer_sizes;
	layer_sizes.push_back(num_inputs);
	layer_sizes.push_back(num_hidden);
	//layer_sizes.push_back(num_hidden2);
	//layer_sizes.push_back(num_hidden3);
	layer_sizes.push_back(num_outputs);

	Improved_Layer::Linked_Layer_Loop_Eval::NeuralNetwork *neuralNet =
		new Improved_Layer::Linked_Layer_Loop_Eval::NeuralNetwork(layer_sizes);








	//	indexes
	//vector< matrix > output_buffer;

	cout << endl;
	cout << "Training, please wait ..." << endl;
	//return;
	timer.Start();

	//	Sleep(2000);

	float last_sum_squared_errors = 0.0f;
	int positive_error_delta_count = 0;
	int negative_error_delta_count = 0;
	int alternation_count = 0;

	float tolerance = 0.05;

	matrix Cost_Matrix(1, 3);

	matrix scores(training_set.size(), 3);
	matrix expect(training_set.size(), 3);
	for (int mm = 0; mm < num_iterations; mm++)
	{

		for (int q = 0; q < training_set.size(); q++)
		{
			// index remap the shuffled set to the original data
			input_matrix(0, 0) = data[training_set[q]][0];
			input_matrix(0, 1) = data[training_set[q]][1];


			matrix expected(1, 3);
			for (int p = 0; p < 3; p++)
			{
				if ((int)data[training_set[q]][2] == p)
				{
					expected(0, p) = 1.0;
				}
				else
				{
					expected(0, p) = 0.0;
				}
			}


			// feed forward


			matrix output = neuralNet->FeedForward_ReLU(input_matrix);

			for (int p = 0; p < 3; p++)
			{
				scores(q, p) = output(0, p);
				expect(q, p) = expected(0, p);
			}

			matrix errors = expected - output;
			//errors = ln(output);// *(-1);
			matrix errors2 = errors;

			Cost_Matrix = Cost_Matrix + pow(errors2, 2);






		}

		scores = neuralNet->Compute_Probabilities(scores);
		matrix probs = scores;
		//cout << "scores: " << scores.NumRows() << ", " << scores.NumCols() << endl;

		scores = neuralNet->Compute_Log_Probabilities(scores, expect);


		float data_loss = neuralNet->Compute_Data_Loss(scores, training_set.size());

		float reg_loss = neuralNet->Compute_Regularization_Loss(reg);

		float total_loss = data_loss + reg_loss;

		if (mm % 20 == 0)
			cout << "total loss: " << total_loss << ", data_loss: " << data_loss << ", reg loss: " << reg_loss << endl;

		for (int q = 0; q < training_set.size(); q++)
		{
			// back propagate output
			matrix probs_(1, 3);
			matrix expected(1, 3);

			for (int i = 0; i < 3; i++)
			{
				probs_(0, i) = probs(q, i);
				expected(0, i) = expect(q, i);

				//if (expected(0, i) == 1.0)
				//{
				//	probs_(0, i) = probs(0, i) - 1;
				//}
			}
			neuralNet->BackPropagateErrors(probs_,expected,training_set.size(), alpha, beta);


			// weight deltas
			neuralNet->ComputeDeltas(probs_, expected, training_set.size(), alpha, beta);
		}


		// update weights
		neuralNet->UpdateWeights(reg);

		//Cost_Matrix.print(3);
		Cost_Matrix.ToZero();



	}
	timer.Update();
	timer.Stop();
	double time_taken = timer.GetTimeDelta();
	cout << "Finished training, Total calculation performed in " << time_taken << " seconds" << endl;

	time_ += time_taken;

	sum_squared_errors = 0.0f; // used here to count the number of correct guesses

	for (int q = 0; q < test_set.size(); q++)
	{


		input_matrix(0, 0) = data[test_set[q]][0];
		input_matrix(0, 1) = data[test_set[q]][1];

		//input_matrix(0, 2) = -1.0f; // bias is always -1



		matrix test_output = neuralNet->FeedForward_ReLU(input_matrix);

		if (test_output.NumCols() != 3) cout << "should have more columns" << endl;

		int actual_type = (int)data[test_set[q]][2];

		int found_type = 0;

		if ((test_output(0, 0) > (1 - tolerance)) && (test_output(0, 1) < tolerance) && (test_output(0, 2) < tolerance))
		{
			found_type = 0;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) > (1 - tolerance)) && (test_output(0, 2) < tolerance))
		{
			found_type = 1;
		}
		if ((test_output(0, 0) < tolerance) && (test_output(0, 1) < tolerance) && (test_output(0, 2) > (1 - tolerance)))
		{
			found_type = 2;
		}

		//	cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}


int main(int argc, char* argv[])
{

	// load the Iris dataset
	CSV iris_data;
	iris_data.test();


	// create a vector of integers to index the iris data array
	vector<int> indexes;
	for (int i = 0; i < iris_data.iris_data.size(); i++)
		indexes.push_back(i);

	// shuffle the indexes to randomize the order of the data
	
	std::random_shuffle(indexes.begin(), indexes.end());
	
	// compute the half size of the data set
	int half_size = indexes.size() / 2;

	// create a vector of indexes for training
	vector<int> training_set;
	// create a vector of indexes for testing
	vector< int > test_set;

	// store the first half of the indexes in the training set
	// and the second half of the indexes in the test set
	for (int i = 0; i < indexes.size(); i++)
	{
		if (i < 100)
		{
			training_set.push_back(indexes[i]);
		}
		else
		{
			test_set.push_back(indexes[i]);
		}
	}
	
	double total_time = 0;
	// test the IRIS dataset
	cout << "===========================================================================================" << endl;
	cout << "Testing Linked_Layer_Loop_Eval neural net object" << endl;

//	Compute_IRIS_data_version_2_<Vanilla_Layer::Linked_Layer_Loop_Eval::NeuralNetwork>
//		(600, iris_data, training_set, test_set,
//		indexes, total_time);

	cout << endl << endl;
	cout << "Finished testing linked layer with iterative evaluation neural net object" << endl;


	// load the spiral dataset
	std::ifstream data_output;


	data_output.open("some.csv", std::ifstream::in);
	// 
	vector<string> tokenized_string;
	copy(istream_iterator<string>(data_output),
		istream_iterator<string>(),
		back_inserter<vector<string> >(tokenized_string));

	vector<vector<string>> output;
	// split lines by commas and store in output
	for (int j = 0; j < tokenized_string.size(); j++) {
		istringstream ss(tokenized_string[j]);
		vector<string> result;
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			result.push_back(substr);
		}
		output.push_back(result);

		for (int c = 0; c < result.size(); c++)
			cout << result[c] << ", ";
		cout << endl;
	}
		
	data_output.close();

	vector< vector <float> > X;
	vector <int>  Y;

	for (int i = 0; i < output.size(); i++)
	{
		
		vector<float> v;
		v.push_back(atof(output[i][0].c_str()));
		v.push_back(atof(output[i][1].c_str()));
		v.push_back(atof(output[i][2].c_str()));

		X.push_back(v);

	}



	vector<int> indexes_spiral;
	for (int i = 0; i < X.size(); i++)
	{
	
		indexes_spiral.push_back(i);
	}

	data_output.close();

	// shuffle the indexes to randomize the order of the data

	for (int i = 0; i < 5; i++)
		std::random_shuffle(indexes_spiral.begin(), indexes_spiral.end());


	// create a vector of indexes for training
	vector<int> training_set_spiral;
	// create a vector of indexes for testing
	vector< int > test_set_spiral;

	// store the first half of the indexes in the training set
	// and the second half of the indexes in the test set
	for (int i = 0; i < indexes_spiral.size(); i++)
	{
		if (i < 200)
		{
			training_set_spiral.push_back(indexes_spiral[i]);
		}
		else
		{
			test_set_spiral.push_back(indexes_spiral[i]);
		}
	}
	
	total_time = 0;
	cout << "===========================================================================================" << endl;
	cout << "Testing Linked_Layer_Loop_Eval neural net object" << endl;

	Compute_IRIS_data_version_3_
		(1500, X, training_set_spiral, test_set_spiral,
		indexes_spiral, total_time);

	cout << endl << endl;
	cout << "Finished testing linked layer with iterative evaluation neural net object" << endl;
	

	/*total_time = 0;
	cout << "===========================================================================================" << endl;
	cout << "Testing Linked_Layer_Loop_Eval neural net object" << endl;

	Compute_IRIS_data_version_4_
		(500, X, training_set_spiral, test_set_spiral,
		indexes_spiral, total_time);

	cout << endl << endl;
	cout << "Finished testing linked layer with iterative evaluation neural net object" << endl;*/
	return 0;
}

