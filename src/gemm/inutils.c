#include "inutils.h"
#include <sys/types.h>

void set_CNN(int col, int cnn_num, char *tmp, cnn_t *cnn) {
  switch(col) {
    case 0:
      cnn[cnn_num].layer  = atoi(tmp);
      break;
    case 1: //M
      cnn[cnn_num].mmin  = atoi(tmp);
      cnn[cnn_num].mmax  = atoi(tmp);
      cnn[cnn_num].mstep = 1;
      break;
    case 2: //N
      cnn[cnn_num].nmin  = atoi(tmp);
      cnn[cnn_num].nmax  = atoi(tmp);
      cnn[cnn_num].nstep = 1;
      break;
  case 3: //K
      cnn[cnn_num].kmin  = atoi(tmp);
      cnn[cnn_num].kmax  = atoi(tmp);
      cnn[cnn_num].kstep = 1;
      break;
  }
}

/*
size_t getline(char **restrict buffer, size_t *restrict size,
                FILE *restrict fp) {
  register int c;
  register char *cs = NULL;

  if (cs == NULL) {
    register int length = 0;
    while ((c = getc(fp)) != EOF) {
      cs = (char *)realloc(cs, ++length+1);
      if ((*(cs + length - 1) = c) == '\n') {
        *(cs + length) = '\0';
        *buffer = cs;
                break;
      }
    }
    return (ssize_t)(*size = length);
  } else {
    while (--(*size) > 0 && (c = getc(fp)) != EOF){
      if ((*cs++ = c) == '\n')
        break;
        }
    *cs = '\0';
  }
  return (ssize_t)(*size=strlen(*buffer));
}
*/
testConfig_t* new_CNN_Test_Config(char * argv[]) {
  FILE *fd_conf = fopen(argv[21], "r"); //open config file
  char line[512];
  const char delimiter[] = "\t";
  char *tmp;
  int col;

  testConfig_t *new_testConfig = (testConfig_t *)malloc(sizeof(testConfig_t));
  int cnn_num;
  
  new_testConfig->tmin   = 0;
  new_testConfig->test   = 0;
  new_testConfig->debug  = 0;
  
  cnn_num=0;    
  while (fgets(line, 512, fd_conf) != NULL)
    if (line[0] != '#') {      
      col = 0;
      tmp = strtok(line, delimiter);
      if (tmp == NULL)
	break;
      set_CNN(col, cnn_num, tmp, new_testConfig->cnn);
      col++;
      for (;;) {
	tmp = strtok(NULL, delimiter);
	if (tmp == NULL)
	  break;
	set_CNN(col, cnn_num, tmp, new_testConfig->cnn);
	col++;
      }

      cnn_num++;
    }

  fclose(fd_conf); 

  new_testConfig->cnn_num = cnn_num;
  
  return new_testConfig;
}

void free_CNN_Test_Config(testConfig_t *testConfig) {
  free(testConfig);
}
