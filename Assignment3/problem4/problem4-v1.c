#define _GNU_SOURCE
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include <linux/time.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, register double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }


  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // for(int i = 0;i<sizeof(b)/sizeof(double);i++){
  //   printf("b[%d] --> %lf,\n",i,b[i]);
  // }

  // grid value initialize
  // initialize value of kk;
  register double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, register double kk) {
  // results values
  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  // constraint values
  double q1 , q2 , q3 , q4 , q5 , q6 , q7 , q8 , q9 , q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v1.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  e1 = kk * ey1;
  e2 = kk * ey2;
  e3 = kk * ey3;
  e4 = kk * ey4;
  e5 = kk * ey5;
  e6 = kk * ey6;
  e7 = kk * ey7;
  e8 = kk * ey8;
  e9 = kk * ey9;
  e10 = kk * ey10;

  x1 = dd1;
  x2 = dd4;
  x3 = dd7;
  x4 = dd10;
  x5 = dd13;
  x6 = dd16;
  x7 = dd19;
  x8 = dd22;
  x9 = dd25;
  x10 = dd28;

  // for loop upper values
  int s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;

  s1 = floor((dd2 - dd1) / dd3);
  s2 = floor((dd5 - dd4) / dd6);
  s3 = floor((dd8 - dd7) / dd9);
  s4 = floor((dd11 - dd10) / dd12);
  s5 = floor((dd14 - dd13) / dd15);
  s6 = floor((dd17 - dd16) / dd18);
  s7 = floor((dd20 - dd19) / dd21);
  s8 = floor((dd23 - dd22) / dd24);
  s9 = floor((dd26 - dd25) / dd27);
  s10 = floor((dd29 - dd28) / dd30);

  // printf("%lf %lf %f, %d\n",dd1,dd2,dd3,s1);

  // grid search starts
  for ( x1 = dd1; x1 < dd2; x1+=dd3) {
    // x1 = dd1 + r1 * dd3;
    double temp11 = -d1+c11*x1;
    double temp12 = -d2+c21*x1;
    double temp13 = -d3+c31*x1;
    double temp14 = -d4+c41*x1;
    double temp15 = -d5+c51*x1;
    double temp16 = -d6+c61*x1;
    double temp17 = -d7+c71*x1;
    double temp18 = -d8+c81*x1;
    double temp19 = -d9+c91*x1;
    double temp110 = -d10+c101*x1;
    if (!((temp11 <=e1) && (temp12 <=e2) && (temp13 <=e3) && (temp14 <=e4) && (temp15 <=e5) && (temp16 <=e6) && (temp17 <=e7) && (temp18 <=e8) && (temp19 <=e9) && (temp110 <=e10))){
      continue;
    }

    for ( x2 = dd4; x2 < dd5; x2+=dd6) {
      // x2 = dd4 + r2 * dd6;
      double temp21 = temp11+c12*x2;
      double temp22 = temp12+c22*x2;
      double temp23 = temp13+c32*x2;
      double temp24 = temp14+c42*x2;
      double temp25 = temp15+c52*x2;
      double temp26 = temp16+c62*x2;
      double temp27 = temp17+c72*x2;
      double temp28 = temp18+c82*x2;
      double temp29 = temp19+c92*x2;
      double temp210 = temp110+c102*x2;
      if (!((temp21 <=e1) && (temp22 <=e2) && (temp23 <=e3) && (temp24 <=e4) && (temp25 <=e5) && (temp26 <=e6) && (temp27 <=e7) && (temp28 <=e8) && (temp29 <=e9) && (temp210 <=e10))){
        continue;
      }

      for ( x3 = dd7; x3 < dd8; x3+=dd9) {
        // x3 = dd7 + r3 * dd9;
        double temp31 = temp21+c13*x3;
        double temp32 = temp22+c23*x3;
        double temp33 = temp23+c33*x3;
        double temp34 = temp24+c43*x3;
        double temp35 = temp25+c53*x3;
        double temp36 = temp26+c63*x3;
        double temp37 = temp27+c73*x3;
        double temp38 = temp28+c83*x3;
        double temp39 = temp29+c93*x3;
        double temp310 = temp210+c103*x3;
        if (!((temp31 <=e1) && (temp32 <=e2) && (temp33 <=e3) && (temp34 <=e4) && (temp35 <=e5) && (temp36 <=e6) && (temp37 <=e7) && (temp38 <=e8) && (temp39 <=e9) && (temp310 <=e10))){
          continue;
        }

        for (x4 = dd10; x4 < dd11; x4+=dd12) {
          // x4 = dd10 + r4 * dd12;
          double temp41 = temp31+c14*x4;
          double temp42 = temp32+c24*x4;
          double temp43 = temp33+c34*x4;
          double temp44 = temp34+c44*x4;
          double temp45 = temp35+c54*x4;
          double temp46 = temp36+c64*x4;
          double temp47 = temp37+c74*x4;
          double temp48 = temp38+c84*x4;
          double temp49 = temp39+c94*x4;
          double temp410 = temp310+c104*x4;
          if (!((temp41 <=e1) && (temp42 <=e2) && (temp43 <=e3) && (temp44 <=e4) && (temp45 <=e5) && (temp46 <=e6) && (temp47 <=e7) && (temp48 <=e8) && (temp49 <=e9) && (temp410 <=e10))){
            continue;
          }

          for (x5 = dd13; x5 < dd14; x5+=dd15) {
            // x5 = dd13 + r5 * dd15;
            double temp51 = temp41+c15*x5;
            double temp52 = temp42+c25*x5;
            double temp53 = temp43+c35*x5;
            double temp54 = temp44+c45*x5;
            double temp55 = temp45+c55*x5;
            double temp56 = temp46+c65*x5;
            double temp57 = temp47+c75*x5;
            double temp58 = temp48+c85*x5;
            double temp59 = temp49+c95*x5;
            double temp510 = temp410+c105*x5;
            if (!((temp51 <=e1) && (temp52 <=e2) && (temp53 <=e3) && (temp54 <=e4) && (temp55 <=e5) && (temp56 <=e6) && (temp57 <=e7) && (temp58 <=e8) && (temp59 <=e9) && (temp510 <=e10))){
              continue;
            }

            for (x6 = dd16; x6 < dd17; x6+=dd18) {
              // x6 = dd16 + r6 * dd18;
              double temp61 = temp51+c16*x6;
              double temp62 = temp52+c26*x6;
              double temp63 = temp53+c36*x6;
              double temp64 = temp54+c46*x6;
              double temp65 = temp55+c56*x6;
              double temp66 = temp56+c66*x6;
              double temp67 = temp57+c76*x6;
              double temp68 = temp58+c86*x6;
              double temp69 = temp59+c96*x6;
              double temp610 = temp510+c106*x6;
              if (!((temp61 <=e1) && (temp62 <=e2) && (temp63 <=e3) && (temp64 <=e4) && (temp65 <=e5) && (temp66 <=e6) && (temp67 <=e7) && (temp68 <=e8) && (temp69 <=e9) && (temp610 <=e10))){
                continue;
              }

              for (x7 = dd19; x7 < dd20; x7+=dd21) {
                // x7 = dd19 + r7 * dd21;
                double temp71 = temp61+c17*x7;
                double temp72 = temp62+c27*x7;
                double temp73 = temp63+c37*x7;
                double temp74 = temp64+c47*x7;
                double temp75 = temp65+c57*x7;
                double temp76 = temp66+c67*x7;
                double temp77 = temp67+c77*x7;
                double temp78 = temp68+c87*x7;
                double temp79 = temp69+c97*x7;
                double temp710 = temp610+c107*x7;
                if (!((temp71 <=e1) && (temp72 <=e2) && (temp73 <=e3) && (temp74 <=e4) && (temp75 <=e5) && (temp76 <=e6) && (temp77 <=e7) && (temp78 <=e8) && (temp79 <=e9) && (temp710 <=e10))){
                  continue;
                }

                for (x8 = dd22; x8 < dd23; x8+=dd24) {
                  // x8 = dd22 + r8 * dd24;
                  double temp81 = temp71+c18*x8;
                  double temp82 = temp72+c28*x8;
                  double temp83 = temp73+c38*x8;
                  double temp84 = temp74+c48*x8;
                  double temp85 = temp75+c58*x8;
                  double temp86 = temp76+c68*x8;
                  double temp87 = temp77+c78*x8;
                  double temp88 = temp78+c88*x8;
                  double temp89 = temp79+c98*x8;
                  double temp810 = temp710+c108*x8;
                  if (!((temp81 <=e1) && (temp82 <=e2) && (temp83 <=e3) && (temp84 <=e4) && (temp85 <=e5) && (temp86 <=e6) && (temp87 <=e7) && (temp88 <=e8) && (temp89 <=e9) && (temp810 <=e10))){
               
                    continue;
                  }

                  for (x9 = dd25; x9 < dd26; x9+=dd27) {
                    // x9 = dd25 + r9 * dd27;
                    double temp91 = temp81+c19*x9;
                    double temp92 = temp82+c29*x9;
                    double temp93 = temp83+c39*x9;
                    double temp94 = temp84+c49*x9;
                    double temp95 = temp85+c59*x9;
                    double temp96 = temp86+c69*x9;
                    double temp97 = temp87+c79*x9;
                    double temp98 = temp88+c89*x9;
                    double temp99 = temp89+c99*x9;
                    double temp910 = temp810+c109*x9;
                    if (!((temp91 <=e1) && (temp92 <=e2) && (temp93 <=e3) && (temp94 <=e4) && (temp95 <=e5) && (temp96 <=e6) && (temp97 <=e7) && (temp98 <=e8) && (temp99 <=e9) && (temp910 <=e10))){
                      continue;
                    }

                    for (x10 = dd28; x10 < dd29; x10+=dd30) {
                      // x10 = dd28 + r10 * dd30;
                      


                      // constraints

                      q1 = fabs(c11 * x1 + c12 * x2 + c13 * x3 + c14 * x4 + c15 * x5 + c16 * x6 +
                                c17 * x7 + c18 * x8 + c19 * x9 + c110 * x10 - d1);

                      q2 = fabs(c21 * x1 + c22 * x2 + c23 * x3 + c24 * x4 + c25 * x5 + c26 * x6 +
                                c27 * x7 + c28 * x8 + c29 * x9 + c210 * x10 - d2);

                      q3 = fabs(c31 * x1 + c32 * x2 + c33 * x3 + c34 * x4 + c35 * x5 + c36 * x6 +
                                c37 * x7 + c38 * x8 + c39 * x9 + c310 * x10 - d3);

                      q4 = fabs(c41 * x1 + c42 * x2 + c43 * x3 + c44 * x4 + c45 * x5 + c46 * x6 +
                                c47 * x7 + c48 * x8 + c49 * x9 + c410 * x10 - d4);
                      q5 = fabs(c51 * x1 + c52 * x2 + c53 * x3 + c54 * x4 + c55 * x5 + c56 * x6 +
                                c57 * x7 + c58 * x8 + c59 * x9 + c510 * x10 - d5);

                      q6 = fabs(c61 * x1 + c62 * x2 + c63 * x3 + c64 * x4 + c65 * x5 + c66 * x6 +
                                c67 * x7 + c68 * x8 + c69 * x9 + c610 * x10 - d6);

                      q7 = fabs(c71 * x1 + c72 * x2 + c73 * x3 + c74 * x4 + c75 * x5 + c76 * x6 +
                                c77 * x7 + c78 * x8 + c79 * x9 + c710 * x10 - d7);

                      q8 = fabs(c81 * x1 + c82 * x2 + c83 * x3 + c84 * x4 + c85 * x5 + c86 * x6 +
                                c87 * x7 + c88 * x8 + c89 * x9 + c810 * x10 - d8);

                      q9 = fabs(c91 * x1 + c92 * x2 + c93 * x3 + c94 * x4 + c95 * x5 + c96 * x6 +
                                c97 * x7 + c98 * x8 + c99 * x9 + c910 * x10 - d9);

                      q10 = fabs(c101 * x1 + c102 * x2 + c103 * x3 + c104 * x4 + c105 * x5 +
                                 c106 * x6 + c107 * x7 + c108 * x8 + c109 * x9 + c1010 * x10 - d10);

                      if ((q1 <= e1) && (q2 <= e2) && (q3 <= e3) && (q4 <= e4) && (q5 <= e5) &&
                          (q6 <= e6) && (q7 <= e7) && (q8 <= e8) && (q9 <= e9) && (q10 <= e10)) {
                        pnts = pnts + 1;

                        // xi's which satisfy the constraints to be written in file
                        fprintf(fptr, "%lf\t", x1);
                        fprintf(fptr, "%lf\t", x2);
                        fprintf(fptr, "%lf\t", x3);
                        fprintf(fptr, "%lf\t", x4);
                        fprintf(fptr, "%lf\t", x5);
                        fprintf(fptr, "%lf\t", x6);
                        fprintf(fptr, "%lf\t", x7);
                        fprintf(fptr, "%lf\t", x8);
                        fprintf(fptr, "%lf\t", x9);
                        fprintf(fptr, "%lf\n", x10);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
