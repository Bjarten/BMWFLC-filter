
//Constants
//#define PI 3.141592654   // π (predefined)
#define Fmin 4    // starting frequency
#define Fmax 7// starting frequency
#define dF 0.3    // ΔF: frequency step (spacing between the dominant components)
#define N (int)(((Fmax-Fmin)/dF) + 1)      // number of dominant frequency components
#define kappa 0.01
#define G 100 // length of sliding window for max amplitude.
#define dT 0.01
#define Tp 10
#define alpha 0.67
#define delta (1/dT)*Tp
#define rho pow(alpha,(1/delta))
#define H  0.001
 
//i.e. 1 Hz to 4 Hz band
//--------------------------------------------------------
//External variables
float v[N];//array of angular frequencies
float x[2][N];//reference input vector, 1st row for sin and 2nd row for cos
float w[2][N];//weight vector, 1st row for sin and 2nd row for cos
float MU = 0.001;    // μ: adaptive gain parameter
float MU0_1 = 0.00000001;
float MU0_2 = 0.00000001;
float learning_rate_scaling_factors[6] = {50, 25, 10, 5, 2, 1};
float peak_1_t_since_found = 0;
float peak_2_t_since_found = 0;



int p1index = -1, p2index = -1, p1indexPrev = -1, p2indexPrev = -1;

float peakAmplitude = 0;
int peakUdateCounter = 0;

float magnitudes[N]; // array of magnitudes
//--------------------------------------------------------
//initialize BMFLC filter
//Get angular velocities and initialize weights
void InitBMWFLC()
{
  int i;
  for (i = 0; i < N; i++) {
    v[i] = 2 * PI * (Fmin + dF * i); //assign a band of frequencies
  }
  for (i = 0; i < N; i++) {
    w[0][i] = 0;  //init weights
    w[1][i] = 0;
  }
}

void find_magnitude_peaks()
{
  float peak1 = 0, peak2 = 0, magDiff = 0, magPrev = 0, magDiffPrev = 0;
  for (int i = 0; i < N; i++)
  {
    float a = w[0][i];
    float b = w[1][i];
    float c = pow(a, 2) + pow(b, 2);
    //float c = abs(a) + abs(b);
    magnitudes[i] = c;
    //Serial.println(c);
  }
  for (int i = 0; i < N; i++)
  {
    magDiff = magnitudes[i] - magPrev;

    if (magDiff < 0 && magDiffPrev > 0)
    {
      if (peak1 < magPrev && p1index != (i-1))
      {
        peak2 = peak1;
        p2index = p1index;
        peak1 = magPrev;
        p1index = i - 1;
      }
      else if (peak2 < magPrev && (i - 1) != p1index)
      {
         peak2 = magPrev;
         p2index = i - 1;
       
      }
    }
    else if ((i == (N - 1)) && (magDiff > 0))
    {
      if (peak1 < magDiff)
      {
        peak2 = peak1;
        p2index = p1index;
        peak1 = magDiff;
        p1index = i; 
      }
    }

    magPrev = magnitudes[i];
    magDiffPrev = magDiff;
  }

  if(p1indexPrev != p1index)
  {
   peak_1_t_since_found = 0;
   peak_2_t_since_found = peak_1_t_since_found;
  }
  if (p2indexPrev != p2index && p2index == p1indexPrev)
  {
      peak_2_t_since_found = peak_1_t_since_found;
  }
  else if (p2indexPrev != p2index && p2index != p1indexPrev)
  {
    peak_2_t_since_found = 0;  
  }

  p1indexPrev = p1index;
  p2indexPrev = p2index;
}

void adaptive_learningrate(float s)
{
   if (peak_1_t_since_found < 5)
      peak_1_t_since_found += 0.01;
   if (peak_2_t_since_found < 5)
      peak_2_t_since_found += 0.01;      

  if (peakAmplitude < s)
  {
    peakAmplitude = s;  
  }
  
  if (peakUdateCounter >= G)
  {
    MU = kappa / peakAmplitude;
    peakUdateCounter = 0;
  }
  //Serial.println(learning_rate_scaling_factors[(int)(peak_1_t_since_found)]);
  //Serial.println(learning_rate_scaling_factors[(int)(peak_2_t_since_found)]);
  MU0_1 = learning_rate_scaling_factors[(int)(peak_1_t_since_found)]*MU * H;
  MU0_2 = learning_rate_scaling_factors[(int)(peak_2_t_since_found)]*MU * H;
  peakUdateCounter++;  
}

//--------------------------------------------------------
//BMFLC filter
//input (k: time instant, s: reference signal)
//output (y: estimated signal)
float BWMFLC(float k, float s)
{

  // adapt learning rate
  adaptive_learningrate(s);
  //-----------------------------------------------------------------

  int i; float err, y, z;
  //-----------------------------------------------------------------
  //find reference input vector
  for (i = 0; i < N; i++) {
    x[0][i] = sin(v[i] * k);
    x[1][i] = cos(v[i] * k);
  }
  //-----------------------------------------------------------------
  //find estimated signal, y
  for (i = 0, y = 0; i < N; i++) {
    y += w[0][i] * x[0][i] + w[1][i] * x[1][i];
  }
  //-----------------------------------------------------------------
  //adapt the weights
  for (i = 0, err = s - y; i < N; i++) {
    w[0][i] = w[0][i] * rho + 2 * MU * x[0][i] * err;
    w[1][i] = w[1][i] * rho + 2 * MU * x[1][i] * err;
  }

  //-----------------------------------------------------------------

  find_magnitude_peaks();

  
  z += (p1index + 1) * (w[0][p1index] * x[1][p1index] - w[1][p1index] * x[0][p1index]);
  v[p1index] = v[p1index] + 2 * MU0_1 * err * z;
  z += (p2index + 1) * (w[0][p2index] * x[1][p2index] - w[1][p2index] * x[0][p2index]);
  v[p2index] = v[p2index] + 2 * MU0_2 * err * z;


}

float get_dominant_frequency_1()
{
  return v[p1index] / (2 * PI);
}

float get_dominant_frequency_2()
{
  return v[p2index] / (2 * PI);
}


//--------------------------------------------------------
