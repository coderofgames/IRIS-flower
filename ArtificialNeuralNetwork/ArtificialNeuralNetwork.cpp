// ArtificialNeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Timer.h"
#include <iostream>
#include <vector>
#include <algorithm>


#include "Utils.h"
#include "my_matrix.h"

#include <math.h>
//#undef matrix;

typedef LINALG::matrixf matrix;



float tan_hyperbolic(float x)
{
	float ex = exp(x);
	float emx = exp(-x);
	return (ex - emx)/ (ex + emx);
}
float tan_hyperbolic_deriv(float y)
{
	return (1 + y)* (1 - y);
}


matrix& tan_hyperbolic(matrix &out, matrix &m)
{
	if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
	{
		out.destroy();
		out = m;
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumCols(); j++)
			out(i, j) = tan_hyperbolic( m(i, j) );

	return out;
}

matrix& tan_hyperbolic_derivative(matrix& out, matrix &m)
{
	if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
	{
		out = m;
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumCols(); j++)
			out(i, j) = tan_hyperbolic_deriv(m(i, j));

	return out;
}

float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

float sigmoid_deriv(float x)
{
	return x * (1 - x);
}

matrix& sigmoid(matrix &out, matrix &m)
{
	if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
	{
		out = m;
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumCols(); j++)
			out(i, j) = 1.f / (1.f + exp(-m(i, j)));

	return out;
}

matrix& sigmoid_deriv(matrix& out, matrix &m)
{
	if (out.NumCols() != m.NumCols() || out.NumRows() != m.NumRows())
	{
		out = m;
	}
	for (int i = 0; i < m.NumRows(); i++)
		for (int j = 0; j < m.NumCols(); j++)
			out(i, j) = m(i, j) * (1 - m(i, j));

	return out;
}


matrix& pow(matrix& source, float exponent)
{
	for (int i = 0; i < source.NumRows(); i++)
	{
		for (int j = 0; j < source.NumCols(); j++)
		{
			source(i, j) = pow(source(i, j), exponent);
		}
	}
	return source;
}

matrix& ln(matrix& source) 
{
	for (int i = 0; i < source.NumRows(); i++)
	{
		for (int j = 0; j < source.NumCols(); j++)
		{
			source(i, j) = log(source(i, j));
		}
	}
	return source;
}

matrix& exp(matrix& source)
{
	for (int i = 0; i < source.NumRows(); i++)
	{
		for (int j = 0; j < source.NumCols(); j++)
		{
			source(i, j) = exp(source(i, j));
		}
	}
	return source;
}

float SUM(matrix& source)
{
	float sum = 0.0f;
	for (int i = 0; i < source.NumRows(); i++)
	{
		for (int j = 0; j < source.NumCols(); j++)
		{
			sum += source(i, j);// = exp(source(i, j));
		}
	}
	return sum;
}


namespace Vanilla_Layer
{
	namespace Non_Lnked_Layer
	{
		class Layer
		{
		public:
			matrix neurons_;
			matrix thetas_;
			matrix weights_;

			matrix delta_thetas_;
			matrix delta_weights_;

			matrix deltas_;

			Layer(){}

			Layer(unsigned int num_elements, unsigned int num_inputs)
			{
				neurons_.create(1, num_elements);
				weights_.create(num_inputs, num_elements);
				thetas_.create(1, num_elements);
				delta_weights_.create(num_inputs, num_elements);
				delta_thetas_.create(1, num_elements);
			}



			void init_random_sample_weights_iris()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) / 5;
					}
				}

				for (int i = 0; i < thetas_.NumRows(); i++)
				{
					for (int j = 0; j < thetas_.NumCols(); j++)
					{
						thetas_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}



			//1 7 ... 7 3   3 3

			void FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * weights_ - thetas_;
				sigmoid(neurons_, neurons_);
#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void BackPropogate(matrix &next_layer_weights_, matrix &next_layer_delta_weights_)
			{
				sigmoid_deriv(deltas_, neurons_);

				next_layer_weights_.transpose();

				// OPERATOR PRECEDENCE! it is easy to forget what you are computing
				deltas_ = deltas_ | (next_layer_delta_weights_ * next_layer_weights_);

				next_layer_weights_.transpose();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				sigmoid_deriv(deltas_, neurons_);

				//	deltas_ = (deltas_)| output_error;

				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(matrix &input_matrix, float alpha, float beta)
			{
				delta_weights_.transpose();

				deltas_.transpose();

				delta_weights_ = delta_weights_ * beta + deltas_* input_matrix * alpha;

				delta_weights_.transpose();

				deltas_.transpose();

				delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;

#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				weights_ = weights_ + delta_weights_;// 

				thetas_ = thetas_ + delta_thetas_;

#ifdef VERBOSE
				PrintWeights();
#endif
			}

			void PrintNeurons()
			{
				cout << "Neurons" << endl;
				neurons_.print(3);
				cout << endl;
			}

			void PrintDeltas()
			{
				cout << "Deltas" << endl;
				deltas_.print(3);
			}
			void PrintDeltaWeights()
			{
				cout << "Delta Weights" << endl;
				delta_weights_.print(3);
			}
			void PrintDeltaThetas()
			{
				cout << "Delta Thetas" << endl;
				delta_thetas_.print(3);
			}

			void PrintWeights()
			{
				cout << "Weights" << endl;
				weights_.print(3);
			}
			};

	}

	namespace Linked_Layer
	{
		class Layer;

		class LayerConnection
		{
		public:
			LayerConnection(Layer* Prev, Layer* Next, int num_inputs, int num_outputs)
			{
				weights_.create(num_inputs, num_outputs);
				delta_weights_.create(num_inputs, num_outputs);

				prev = Prev;
				next = Next;
			}

			void init_random_sample_weights_iris()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}

			void Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta);

			void UpdateWeights();

			Layer * prev;
			Layer * next;

			matrix weights_;
			matrix delta_weights_;
		};

		

		class Layer
		{
		public:
			LayerConnection *connection_in;
			LayerConnection *connection_out;

			matrix neurons_;
			matrix thetas_;
			matrix weights_;

			matrix delta_thetas_;
			matrix delta_weights_;

			matrix deltas_;

			Layer(){}

			Layer(unsigned int num_elements, unsigned int num_inputs)
			{
				neurons_.create(1, num_elements);
				weights_.create(num_inputs, num_elements);
				thetas_.create(1, num_elements);
				delta_weights_.create(num_inputs, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			Layer(unsigned int num_elements, unsigned int num_inputs, LayerConnection *in, LayerConnection *out) 
			{
				connection_in = in;
				connection_out = out;
				 
				neurons_.create(1, num_elements);
				weights_.create(num_inputs, num_elements);
				thetas_.create(1, num_elements);
				delta_weights_.create(num_inputs, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			void init_random_sample_weights_iris()
			{
				for (int i = 0; i < weights_.NumRows(); i++)
				{
					for (int j = 0; j < weights_.NumCols(); j++)
					{
						weights_(i, j) = RandomFloat(0, 1) / 5;
					}
				}

				for (int i = 0; i < thetas_.NumRows(); i++)
				{
					for (int j = 0; j < thetas_.NumCols(); j++)
					{
						thetas_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}

			matrix& FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;
				//neurons_ = input_matrix * weights_ - thetas_;
				sigmoid(neurons_, neurons_);

				if (connection_out && connection_out->next )
					return connection_out->next->FeedForward(neurons_);

				return neurons_;
#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void BackPropogate()
			{
				sigmoid_deriv(deltas_, neurons_);

				//if ( connection_out )
				
				connection_out->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->weights_);
				connection_out->weights_.transpose();
				
				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();

				//next_layer_weights_.transpose();

				// OPERATOR PRECEDENCE! it is easy to forget what you are computing
				//deltas_ = deltas_ | (next_layer_delta_weights_ * next_layer_weights_);

				//next_layer_weights_.transpose();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				sigmoid_deriv(deltas_, neurons_);

				//	deltas_ = (deltas_)| output_error;

				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas( float alpha, float beta)
			{
				if ( connection_in )
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);
				//delta_weights_.transpose();

				//deltas_.transpose();

				//delta_weights_ = delta_weights_ * beta + deltas_* input_matrix * alpha;

				//delta_weights_.transpose();

				//deltas_.transpose();

				//delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;
				delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;


#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				if (connection_in)
					connection_in->UpdateWeights();

				thetas_ = thetas_ + delta_thetas_;

#ifdef VERBOSE
				PrintWeights();
#endif
			}

			void PrintNeurons()
			{
				cout << "Neurons" << endl;
				neurons_.print(3);
				cout << endl;
			}

			void PrintDeltas()
			{
				cout << "Deltas" << endl;
				deltas_.print(3);
			}
			void PrintDeltaWeights()
			{
				cout << "Delta Weights" << endl;
				delta_weights_.print(3);
			}
			void PrintDeltaThetas()
			{
				cout << "Delta Thetas" << endl;
				delta_thetas_.print(3);
			}

			void PrintWeights()
			{
				cout << "Weights" << endl;
				weights_.print(3);
			}
		};


		void LayerConnection::Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta)
		{
			delta_weights_.transpose();

			deltas_.transpose();

			delta_weights_ = delta_weights_ * beta + deltas_* this->prev->neurons_ * alpha;

			delta_weights_.transpose();

			deltas_.transpose();


			if (prev)
				prev->ComputeWeightDeltas(alpha, beta);


		}

		void LayerConnection::UpdateWeights()
		{
			weights_ = weights_ + delta_weights_;// 

			if (prev)
				prev->UpdateWeights();
		}

		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork( vector<int> &layer_sizes ){
				

				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];
					
					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++ )
					{
													// num elements  num outputs
						input_layer[i] = new Layer( layer_sizes[i], layer_sizes[i-1] ) ;

						input_layer[i]->connection_in = new LayerConnection( input_layer[i - 1],  // prev
																			 input_layer[i],      // next (this one)
																			 layer_sizes[i - 1], // num_inputs (nodes in prev)
																			 layer_sizes[i]);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this
						
					}

					input_layer[num_layers - 1]->connection_out = 0; // or something ...

					// initialize the weights
					for (int i = 0; i < num_layers; i++)
					{
						if (input_layer[i]->connection_in)
							input_layer[i]->connection_in->init_random_sample_weights_iris();
					}

				}
			}

			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;
				return input_layer[1]->FeedForward(input);
			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers-1]->BackPropogate_output(errors);
			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();
			}

			int num_layers=0;
			Layer **input_layer;
			
		};
	}
	

	
	

	
}


namespace Optimized_Layer
{

	//#define VERBOSE
	class Layer
	{
	public:
		matrix neurons_;
		matrix thetas_;
		matrix weights_;

		matrix delta_thetas_;
		matrix delta_weights_;

		matrix deltas_;

		Layer(){}

		Layer(unsigned int num_elements, unsigned int num_inputs)
		{


			neurons_.create(1, num_elements);


			weights_.create(num_elements, num_inputs);


			thetas_.create(1, num_elements);

	
			delta_weights_.create(num_inputs, num_elements);

	
			delta_thetas_.create(1, num_elements);
		}



		void init_random_sample_weights_iris()
		{
			for (int i = 0; i < weights_.NumRows(); i++)
			{
				for (int j = 0; j < weights_.NumCols(); j++)
				{
					weights_(i, j) = RandomFloat(0, 1) / 5;
				}
			}

			for (int i = 0; i < thetas_.NumRows(); i++)
			{
				for (int j = 0; j < thetas_.NumCols(); j++)
				{
					thetas_(i, j) = RandomFloat(0, 1) / 5;
				}
			}
		}



		//1 7 ... 7 3   3 3

		void FeedForward(matrix &input_matrix)
		{
			//weights_.transpose();
			neurons_ = input_matrix.mul_by_transpose(weights_) - thetas_;
			sigmoid(neurons_, neurons_);
		}

		/**
		This is fancy, the output from the softmax function can be thought
		of as a probability distribution : see NeuralNetworks and Deep learning by
		Michael Nielson
		*/
		void FeedForward_Softmax(matrix &input_matrix)
		{
			//weights_.transpose();
			neurons_ = input_matrix.mul_by_transpose(weights_) - thetas_;

			neurons_ = exp(neurons_);
			float sum = SUM(neurons_);
			neurons_ = neurons_ / sum;
		}

		matrix& log_likliehood_cost_softmax(matrix& out)
		{
			out.ToZero();
			return ln(out);
		}

		void BackPropogate(matrix &next_layer_weights_, matrix &next_layer_delta_weights_)
		{
			sigmoid_deriv(deltas_, neurons_);

			//	next_layer_weights_.transpose();



			// OPERATOR PRECEDENCE! it is easy to forget what you are computing
			deltas_ = deltas_ | (next_layer_delta_weights_ * next_layer_weights_);




			//	next_layer_weights_.transpose();

		}

		void BackPropogate_output(matrix& output_error)
		{
			sigmoid_deriv(deltas_, neurons_);

			//	deltas_ = (deltas_)| output_error;

			deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

		}

		void ComputeWeightDeltas(matrix &input_matrix, float alpha, float beta)
		{
			//	delta_weights_.transpose();

			//	deltas_.transpose();

			delta_weights_ = (delta_weights_ * beta).add_transposed(deltas_.mul_transposed(input_matrix)* alpha);

			delta_weights_.transpose();

			//	deltas_.transpose();

			delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;
		}


		void UpdateWeights()
		{
			weights_ = delta_weights_.add_transposed(weights_);// 

			thetas_ = thetas_ + delta_thetas_;
		}



		void PrintNeurons()
		{
			cout << "Neurons" << endl;
			neurons_.print(4);
			cout << endl;
		}

		void PrintDeltas()
		{
			cout << "Deltas" << endl;
			deltas_.print(4);
		}
		void PrintDeltaWeights()
		{
			cout << "Delta Weights" << endl;
			delta_weights_.print(4);
		}
		void PrintDeltaThetas()
		{
			cout << "Delta Thetas" << endl;
			delta_thetas_.print(4);
		}

		void PrintWeights()
		{
			cout << "Weights" << endl;
			weights_.print(4);
		}
	};
}



matrix& QuadraticCostFunction(matrix& out, matrix& desired_outputs, matrix& outputs)
{
	out = pow((desired_outputs - outputs),2) *0.5 ;

	return out;
}




void Compute_IRIS_data_version_2_(int num_iterations, CSV &iris_data, vector<int> &training_set, vector<int> &test_set, vector<int> &indexes)
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

	vector<int> layer_sizes;
	layer_sizes.push_back(num_inputs);
	layer_sizes.push_back(num_hidden);
	layer_sizes.push_back(num_hidden);
	layer_sizes.push_back(num_hidden_2);

	Vanilla_Layer::Linked_Layer::NeuralNetwork *neuralNet = new Vanilla_Layer::Linked_Layer::NeuralNetwork(layer_sizes);








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



	for (int p = 0; p < num_iterations; p++)
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
			

			matrix output = neuralNet->FeedForward(input_matrix);
			
			matrix errors = expected - output;

			for (int p = 0; p < errors.NumCols(); p++)
			{
				if (abs(errors(0, p)) > 0.2)
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

			cout << "Test Sample: " << q << ", Found Type: " << found_type << ", Actual Type: " << actual_type << endl;

		if (found_type == actual_type) sum_squared_errors += 1.0f;
	}

	cout << "Finished Test, Total classified correctly of " << test_set.size() << " tested: " << (int)sum_squared_errors << endl;

}


// abs(expected - y)
//  
// y(1-y)^2  = (1-y) * sigmoid_deriv(y) =>  (expected - y) * ssigmoid_deriv(y)
// -y^2(1-y) = -y * sigmoid_deriv(y) = > (expected - y ) * sigmoid_deriv(y)

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



	for (int p = 0; p < num_iterations; p++)
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

	

	for (int p = 0; p < num_iterations; p++)
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

int main(int argc, char* argv[])
{
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


/*	Compute_IRIS_data_version_1_< Optimized_Layer::Layer >(1000, iris_data, training_set, test_set, indexes);

	cout << "completed Deep calculation with 2 hidden layers (Optimized_Layer)" << endl;

	Compute_IRIS_data_version_0_< Optimized_Layer::Layer >(1000, iris_data, training_set, test_set, indexes);

	cout << "completed Shallow calculation (Optimized_Layer)" << endl;
	cout << endl << endl;

	Compute_IRIS_data_version_1_< Vanilla_Layer::Non_Lnked_Layer::Layer >(1000, iris_data, training_set, test_set, indexes);

	cout << "completed Deep calculation with 2 hidden layers (Vanilla)" << endl;
	cout << endl << endl;
*/
	Compute_IRIS_data_version_0_< Vanilla_Layer::Non_Lnked_Layer::Layer >(1000, iris_data, training_set, test_set, indexes);

	cout << "completed Shallow calculation (Vanilla)" << endl;

	



	
	cout << "Testing linked layer neural net object" << endl;

	Compute_IRIS_data_version_2_(500, iris_data, training_set, test_set, indexes);
	cout << endl << endl;
	cout << "Finished testing linked layer neural net object" << endl;
	cout << endl << endl;
	
	return 0;
}

