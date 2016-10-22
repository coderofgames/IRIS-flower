#ifndef NEURAL_NETWORKS_H
#define NEURAL_NETWORKS_H

#include "Utils.h"
#include "my_matrix.h"

#include <math.h>
//#undef matrix;

typedef LINALG::matrixf matrix;

namespace
{
	float tan_hyperbolic(float x)
	{
		float ex = std::exp(x);
		float emx = std::exp(-x);
		return (ex - emx) / (ex + emx);
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
				out(i, j) = tan_hyperbolic(m(i, j));

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
		return 1.f / (1.f + std::exp(-x));
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
				out(i, j) = 1.f / (1.f + std::exp(-m(i, j)));

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
				source(i, j) = std::pow(source(i, j), exponent);
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
				source(i, j) = std::exp(source(i, j));
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









	matrix& QuadraticCostFunction(matrix& out, matrix& desired_outputs, matrix& outputs)
	{
		out = pow((desired_outputs - outputs), 2) *0.5;

		return out;
	}
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


	namespace Linked_Layer_Connection_Matrix_In_Layer
	{
		class Layer;

		class LayerConnection
		{
		public:
			LayerConnection(Layer* Prev, Layer* Next)
			{

				prev = Prev;
				next = Next;
			}


			Layer * prev;
			Layer * next;


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
				neurons_ = input_matrix * weights_ - thetas_;

				sigmoid(neurons_, neurons_);

				if (connection_out && connection_out->next)
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

				connection_out->next->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->next->weights_);
				connection_out->next->weights_.transpose();

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in && connection_in->prev)
				{
					delta_weights_.transpose();

					deltas_.transpose();

					delta_weights_ = delta_weights_ * beta + deltas_* connection_in->prev->neurons_ * alpha;

					delta_weights_.transpose();

					deltas_.transpose();

					delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;

					connection_in->prev->ComputeWeightDeltas(alpha, beta);
				}



#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				if (connection_in && connection_in->prev)
				{
					weights_ = weights_ + delta_weights_;// 

					thetas_ = thetas_ + delta_thetas_;

					connection_in->prev->UpdateWeights();
				}
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




		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
							input_layer[i]);     // next (this one)
						//layer_sizes[i - 1], // num_inputs (nodes in prev)
						//layer_sizes[i]);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this

					}

					input_layer[num_layers - 1]->connection_out = 0; // or something ...

					// initialize the weights
					for (int i = 0; i < num_layers; i++)
					{
						if (input_layer[i]->connection_in)
							input_layer[i]->init_random_sample_weights_iris();
					}

				}
			}

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;
				return input_layer[1]->FeedForward(input);
			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);
			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();
			}

			int num_layers = 0;
			Layer **input_layer;

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

				sigmoid(neurons_, neurons_);

				if (connection_out && connection_out->next)
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

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);

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

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
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

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;
				return input_layer[1]->FeedForward(input);
			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);
			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();
			}

			int num_layers = 0;
			Layer **input_layer;

		};
	}



	namespace Linked_Layer_Loop_Eval
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

			void FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				sigmoid(neurons_, neurons_);

				//if (connection_out && connection_out->next)
				//	return connection_out->next->FeedForward(neurons_);

				//return neurons_;
#ifdef VERBOSE
				PrintNeurons();
#endif
			}
			void FeedForward_Softmax(matrix &input_matrix)
			{
				//weights_.transpose();
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				neurons_ = exp(neurons_);
				float sum = SUM(neurons_);
				neurons_ = neurons_ / sum;
			}
			void BackPropogate()
			{
				sigmoid_deriv(deltas_, neurons_);

				//if ( connection_out )

				connection_out->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->weights_);
				connection_out->weights_.transpose();

				//if (connection_in && connection_in->prev)
				//	connection_in->prev->BackPropogate();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				//if (connection_in && connection_in->prev)
				//	connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);

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


			//	if (prev)
			//		prev->ComputeWeightDeltas(alpha, beta);


		}

		void LayerConnection::UpdateWeights()
		{
			weights_ = weights_ + delta_weights_;// 

			//	if (prev)
			//		prev->UpdateWeights();
		}

		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
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

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;

				for (int i = 1; i < num_layers - 1; i++)
				{
					input_layer[i]->FeedForward(input_layer[i - 1]->neurons_);
				}
				input_layer[num_layers - 1]->FeedForward_Softmax(input_layer[num_layers - 2]->neurons_);

				return input_layer[num_layers - 1]->neurons_;

			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->ComputeWeightDeltas(alpha, beta);
				}
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->UpdateWeights();
				}
			}



			int num_layers = 0;
			Layer **input_layer;

		};
	}



	namespace Linked_Layer_Loop_Eval_Connection_Matrix_In_Layer
	{
		class Layer;

		class LayerConnection
		{
		public:
			LayerConnection(Layer* Prev, Layer* Next)
			{

				prev = Prev;
				next = Next;
			}


			Layer * prev;
			Layer * next;


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

			void FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * weights_ - thetas_;

				sigmoid(neurons_, neurons_);

				//if (connection_out && connection_out->next)
				//	return connection_out->next->FeedForward(neurons_);

				//return neurons_;
#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void BackPropogate()
			{
				sigmoid_deriv(deltas_, neurons_);

				//if ( connection_out )

				connection_out->next->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->next->weights_);
				connection_out->next->weights_.transpose();

				//	if (connection_in && connection_in->prev)
				//		connection_in->prev->BackPropogate();

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				//	if (connection_in && connection_in->prev)
				//		connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in && connection_in->prev)
				{
					delta_weights_.transpose();

					deltas_.transpose();

					delta_weights_ = delta_weights_ * beta + deltas_* connection_in->prev->neurons_ * alpha;

					delta_weights_.transpose();

					deltas_.transpose();

					delta_thetas_ = delta_thetas_ * beta + deltas_* (-1.0f) * alpha;

					//	connection_in->prev->ComputeWeightDeltas(alpha, beta);
				}



#ifdef VERBOSE
				PrintDeltaWeights();
#endif
			}


			void UpdateWeights()
			{
				if (connection_in && connection_in->prev)
				{
					weights_ = weights_ + delta_weights_;// 

					thetas_ = thetas_ + delta_thetas_;

					//	connection_in->prev->UpdateWeights();
				}
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




		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
							input_layer[i]);      // next (this one)
						//layer_sizes[i - 1], // num_inputs (nodes in prev)
						//layer_sizes[i]);    // num_outputs (nodes in this)

						input_layer[i - 1]->connection_out = input_layer[i]->connection_in; // the last layers output connection is the input to this

					}

					input_layer[num_layers - 1]->connection_out = 0; // or something ...

					// initialize the weights
					for (int i = 0; i < num_layers; i++)
					{
						if (input_layer[i]->connection_in)
							input_layer[i]->init_random_sample_weights_iris();
					}

				}
			}

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;

				for (int i = 1; i < num_layers; i++)
				{
					input_layer[i]->FeedForward(input_layer[i - 1]->neurons_);
				}

				return input_layer[num_layers - 1]->neurons_;

			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->ComputeWeightDeltas(alpha, beta);
				}
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->UpdateWeights();
				}
			}



			int num_layers = 0;
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


namespace Improved_Layer
{
	namespace Cost_Functions
	{
		
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

				sigmoid(neurons_, neurons_);

				if (connection_out && connection_out->next)
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

#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

				if (connection_in && connection_in->prev)
					connection_in->prev->BackPropogate();
#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);

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

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
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

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;
				return input_layer[1]->FeedForward(input);
			}

			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);
			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();
			}

			int num_layers = 0;
			Layer **input_layer;

		};
	}



	namespace Linked_Layer_Loop_Eval
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
			matrix delta_thetas_;
			matrix deltas_;

			bool b_ReLU = false;

			Layer(){}

			Layer(unsigned int num_elements, unsigned int num_inputs)
			{
				neurons_.create(1, num_elements);
				thetas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			Layer(unsigned int num_elements, unsigned int num_inputs, LayerConnection *in, LayerConnection *out)
			{
				connection_in = in;
				connection_out = out;

				neurons_.create(1, num_elements);
				thetas_.create(1, num_elements);
				delta_thetas_.create(1, num_elements);
			}

			void init_random_sample_weights_iris()
			{
				for (int i = 0; i < thetas_.NumRows(); i++)
				{
					for (int j = 0; j < thetas_.NumCols(); j++)
					{
						thetas_(i, j) = RandomFloat(0, 1) / 5;
					}
				}
			}

			void FeedForward(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				sigmoid(neurons_, neurons_);


#ifdef VERBOSE
				PrintNeurons();
#endif
			}

			void FeedForward_Dot(matrix &input_matrix)
			{
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				//sigmoid(neurons_, neurons_);


#ifdef VERBOSE
				PrintNeurons();
#endif
			}


			void FeedForward_Softmax(matrix &input_matrix)
			{
				//weights_.transpose();
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				neurons_ = exp(neurons_);
				float sum = SUM(neurons_);
				neurons_ = neurons_ / sum;
			}

			void FeedForward_ReLU(matrix &input_matrix)
			{
				//weights_.transpose();
				neurons_ = input_matrix * connection_in->weights_ - thetas_;

				for (int i = 0; i < neurons_.NumRows(); i++)
				{
					for (int j = 0; j < neurons_.NumCols(); j++)
					{
						neurons_(i, j) = max(neurons_(i, j), 0);
					}
				}
			}

			void BackPropogate()
			{
				if (!this->b_ReLU)
					sigmoid_deriv(deltas_, neurons_);
				else
					deltas_ = neurons_;

				connection_out->weights_.transpose();
				deltas_ = deltas_ | (connection_out->next->deltas_ * connection_out->weights_);
				connection_out->weights_.transpose();

				if (this->b_ReLU)
				{
					for (int r = 0; r < neurons_.NumRows(); r++)
					{
						for (int c = 0; c < neurons_.NumCols(); c++)
						{
							if (neurons_(r, c) <= 0 )
							{
								deltas_(r, c) = 0;
							}
							
						}
					}
				}
				

				//if ( connection_out )




#ifdef VERBOSE
				PrintDeltas();
#endif

			}

			void BackPropogate_output(matrix& output_error)
			{
				deltas_ = output_error; // THIS WORKS FOR THE IRIS CASE

#ifdef VERBOSE
				PrintDeltas();
#endif
			}

			void ComputeWeightDeltas(float alpha, float beta)
			{
				if (connection_in)
					connection_in->Compute_Weight_Deltas(deltas_, alpha, beta);

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

			void PrintDeltaThetas()
			{
				cout << "Delta Thetas" << endl;
				delta_thetas_.print(3);
			}

			void PrintWeights()
			{
				cout << "Delta Thetas" << endl;
				if(connection_in)
					connection_in->weights_.print(3);
			}
			void PrintDeltaWeights()
			{
				cout << "Delta Thetas" << endl;
				if (connection_in)
					connection_in->delta_weights_.print(3);
			}

		};


		void LayerConnection::Compute_Weight_Deltas(matrix &deltas_, float alpha, float beta)
		{
			delta_weights_.transpose();

			deltas_.transpose();

			delta_weights_ = delta_weights_ * beta + deltas_* this->prev->neurons_ * alpha;

			delta_weights_.transpose();

			deltas_.transpose();
		}

		void LayerConnection::UpdateWeights()
		{
			weights_ = weights_ + delta_weights_;// 
		}

		class NeuralNetwork
		{
		public:
			NeuralNetwork(){
				input_layer = 0;
			}

			NeuralNetwork(vector<int> &layer_sizes){


				num_layers = layer_sizes.size();
				if (num_layers > 0)
				{
					input_layer = new Layer*[num_layers];

					input_layer[0] = new Layer(layer_sizes[0], 1);
					input_layer[0]->connection_in = 0;
					for (int i = 1; i < num_layers; i++)
					{
						input_layer[i] = new Layer(layer_sizes[i], layer_sizes[i - 1]);

						input_layer[i]->connection_in =
							new LayerConnection(input_layer[i - 1],  // prev
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

			~NeuralNetwork()
			{
				// initialize the weights
				for (int i = 0; i < num_layers; i++)
				{
					if (input_layer[i]->connection_in)
						delete input_layer[i]->connection_in;
					delete input_layer[i];
				}
				delete[] input_layer;
			}


			matrix& FeedForward(matrix& input)
			{
				input_layer[0]->neurons_ = input;

				for (int i = 1; i < num_layers - 1; i++)
				{
					input_layer[i]->FeedForward(input_layer[i - 1]->neurons_);
				}
				input_layer[num_layers - 1]->FeedForward_Softmax(input_layer[num_layers - 2]->neurons_);

				return input_layer[num_layers - 1]->neurons_;

			}

			matrix& FeedForward_ReLU(matrix& input)
			{
				input_layer[0]->neurons_ = input;

				for (int i = 1; i < num_layers - 1; i++)
				{
					input_layer[i]->b_ReLU = true;
					input_layer[i]->FeedForward_ReLU(input_layer[i - 1]->neurons_);
				}
				input_layer[num_layers - 1]->FeedForward_Dot(input_layer[num_layers - 2]->neurons_);

				return input_layer[num_layers - 1]->neurons_;

			}

			matrix& Compute_Probabilities(matrix& scores)
			{
				scores = exp(scores);
				float sum = SUM(scores);
				if ( sum != 0.0 )
					return scores / sum;
				else return scores;
				//return scores;
			}

			matrix& Compute_Log_Probabilities(matrix& probs, matrix& expected)
			{
				
				for (int i = 0; i < probs.NumRows(); i++)
				{
					for (int j = 0; j < probs.NumCols(); j++)
					{
						if (expected(i, j) == 1.0)
						{
							probs(i, j) = -log(probs(i, j));
						}
					}
				}
				return probs;
			}

			float Compute_Data_Loss(matrix& logProbs, int num_samples)
			{
				return SUM(logProbs) / (float)num_samples;
			}

			float Compute_Regularization_Loss(float reg)
			{
				float reg_loss = 0.0;
				for (int i = 1; i < num_layers; i++)
				{
					reg_loss += SUM(pow(input_layer[i]->connection_in->weights_, 2)) * 0.5 * reg;
				}
				return reg_loss;
			}

			float Compute_Total_Loss(matrix &scores, matrix& expected, float reg, int num_samples)
			{
				return Compute_Data_Loss(
						Compute_Log_Probabilities(
							Compute_Probabilities(scores),
								expected),
									num_samples) + Compute_Regularization_Loss(reg);
			}

			matrix& Compute_Deltas(matrix& probs, matrix& expected, int num_examples)
			{

				for (int i = 0; i < probs.NumRows(); i++)
				{
					for (int j = 0; j < probs.NumCols(); j++)
					{
						if (expected(i, j) == 1.0)
						{
							probs(i, j) = probs(i, j) - 1.0;
						}
					}
				}
				return probs / (float)num_examples;
			}

			void ComputeDeltas(matrix& probs, matrix& expected, int num_examples, float alpha, float beta)
			{
				
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->ComputeWeightDeltas(alpha, beta);
				}
			}
			
			
			void BackPropagateErrors(matrix& probs, matrix& expected, int num_examples, float alpha, float beta)
			{
				input_layer[num_layers - 1]->deltas_ = Compute_Deltas(probs, expected, num_examples);

				input_layer[num_layers - 1]->BackPropogate_output(input_layer[num_layers - 1]->deltas_);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}
			
			void BackPropagateErrors(matrix& errors)
			{
				input_layer[num_layers - 1]->BackPropogate_output(errors);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->BackPropogate();
				}

			}

			void ComputeDeltas(float alpha, float beta)
			{
				input_layer[num_layers - 1]->ComputeWeightDeltas(alpha, beta);

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->ComputeWeightDeltas(alpha, beta);
				}
			}

			void UpdateWeights()
			{
				input_layer[num_layers - 1]->UpdateWeights();

				for (int i = num_layers - 2; i > -1; i--)
				{
					input_layer[i]->UpdateWeights();
				}
			}

			void UpdateWeights(float reg)
			{
				input_layer[num_layers - 1]->connection_in->delta_weights_ = input_layer[num_layers - 1]->connection_in->delta_weights_ * reg;
				input_layer[num_layers - 1]->UpdateWeights();

				for (int i = num_layers - 2; i > 0; i--)
				{
					input_layer[i]->connection_in->delta_weights_ = input_layer[i]->connection_in->delta_weights_ * reg;
					input_layer[i]->UpdateWeights();
				}
			}



			int num_layers = 0;
			Layer **input_layer;

		};
	}




}

/*
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

				if (expected(0, i) == 1.0)
				{
					probs_(0, i) = probs(0, i) - 1;
				}
			}
			neuralNet->BackPropagateErrors(probs_);


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



		matrix test_output = neuralNet->FeedForward(input_matrix);

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
*/
#endif