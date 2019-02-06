#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define NUM_LEARN 10000
#define NUM_SAMPLE 6
#define NUM_INPUT 3
#define NUM_HIDDEN 5
#define NUM_OUTPUT 1
#define EPSILON 0.05
#define THRESHOLD_ERROR 0.01

int tx[NUM_SAMPLE][NUM_INPUT], ty[NUM_SAMPLE][NUM_OUTPUT];
float x[NUM_INPUT + 1],h[NUM_HIDDEN + 1],y[NUM_OUTPUT];
float w1[NUM_INPUT + 1][NUM_HIDDEN],w2[NUM_HIDDEN + 1][NUM_OUTPUT];
float h_back[NUM_HIDDEN + 1], y_back[NUM_OUTPUT];
float sigmoid_table[100];												

int main()
{
	int ilearn, isample, i, j;
	float net_input, error, max_error, epsilon, seed;
	int inet_input;
	FILE *stream;
	
	epsilon = (float)EPSILON;
	
	for ( i = 0; i < 100; i++)
	{
		sigmoid_table[i] = (float)(1.0 / (1.0 + exp((double)(i-50) * -0.2)));
	}
	
	stream = fopen("training.dat","r");
	if (stream == NULL)
	{
		printf("ファイルtraining.datをオープンできません.\n");
		exit(0);
	}
	else
	{
		for ( isample = 0 ;isample < NUM_SAMPLE; isample++)
		{
			for( i = 0; i < NUM_INPUT; i++)
			{
				fscanf( stream,"%d",&tx[isample][i]);
			}
			
			for( i = 0; i < NUM_OUTPUT; i++)
			{
				fscanf( stream,"%d",&ty[isample][i]);
			}
		}

		for ( isample = 0 ;isample < NUM_SAMPLE; isample++)
		{
			printf("訓練データ　NO. %d :  ", isample+1);
			printf("入力: ");
			for ( i = 0; i < NUM_INPUT; i++)
			{
				printf(" %d ", tx[isample][i]);
			}
			printf("出力: ");
			for ( i = 0; i < NUM_OUTPUT; i++)
			{
				printf(" %d ", ty[isample][i]);
			}
			printf("\n");
		}
		fclose(stream);
	}

	seed = (float) 1;
	for (i = 0; i < NUM_INPUT+1; i++)
	{
		for (j = 0; j < NUM_HIDDEN; j++)
		{
			seed = seed * (float)-1;
			w1[i][j] = seed;
		}
	}
	
	for (i = 0; i < NUM_HIDDEN+1; i++)
	{
		for (j = 0; j < NUM_OUTPUT; j++)
		{
			seed = seed * (float)-1;
			w2[i][j] = seed;
		}
	}
	
	for (ilearn = 0; ilearn < NUM_LEARN; ilearn++)
	{
		max_error = 0;
		for (isample = 0; isample < NUM_SAMPLE; isample++)
		{
			for (i = 0; i < NUM_INPUT+1; i++)
			{
				x[i] = tx[isample][i];
			}
			
			x[NUM_INPUT] = (float)1.0;
			for (j = 0; j < NUM_HIDDEN; j++)
			{
				net_input = 0;
				for (i = 0; i < NUM_INPUT+1; i++)
				{
					net_input = net_input + w1[i][j]*x[i];
				}
				inet_input = (int)(net_input * 5) + 50;
				if ( inet_input > 99) inet_input = 99;
				else if (inet_input < 0) inet_input = 0;
				h[j] = sigmoid_table[inet_input];
			}

                        h[NUM_HIDDEN] = (float)1.0;
			for (j = 0; j < NUM_OUTPUT; j++)
			{
				net_input = 0;
				for (i = 0; i < NUM_HIDDEN+1; i++)
				{
					net_input = net_input + w2[i][j]*h[i];
				}
				
				inet_input = (int)(net_input * 5) + 50;
				if ( inet_input > 99) inet_input = 99;
				else if (inet_input < 0) inet_input = 0;
				y[j] = sigmoid_table[inet_input];
			}
			
			error = 0;
			
			for (j = 0; j < NUM_OUTPUT; j++)
			{
				error = error + (ty[isample][j] - y[j])*(ty[isample][j] - y[j]);
			}
			error = error / (float)NUM_OUTPUT;
			if(error > max_error) max_error = error;
			
			printf("学習回数 = %d, 訓練データ NO. = %d, 誤差 = %f \n", ilearn, isample+1,error);

			for (j = 0; j < NUM_OUTPUT; j++)
			{
				y_back[j] = (y[j] - ty[isample][j]) * ((float)1.0 - y[j]) * y[j];
			}
			
			for (i = 0; i < NUM_HIDDEN; i++)
			{
				net_input = 0;
				for (j = 0; j < NUM_OUTPUT; j++)
				{
					net_input = net_input + w2[i][j]*y_back[j];
				}
				h_back[i] = net_input*((float)1.0 - h[i])*h[i];
			}

			for (i = 0; i < NUM_INPUT+1; i++)
			{
				for (j = 0; j < NUM_HIDDEN; j++)
				{
					w1[i][j] = w1[i][j] - epsilon*x[i]*h_back[j];
				}
			}
			
			for (i = 0; i < NUM_HIDDEN+1; i++)
			{
				for (j = 0; j < NUM_OUTPUT; j++)
				{
					w2[i][j] = w2[i][j] - epsilon*h[i]*y_back[j];
				}
			}
		}
		if (max_error < THRESHOLD_ERROR) break;
	}
}

