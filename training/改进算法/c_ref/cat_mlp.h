/* cat_mlp.h - MLP推理(纯C99), 权重从CSV运行时加载
 * CSV布局: W{层}_{out}x{in}.csv 行主序, 每行一行
 * 推理: x'=(x-mean)/scale -> relu(x'W1+b1) -> relu(W2+b2) -> W3+b3 -> softmax
 */
#ifndef CAT_MLP_H
#define CAT_MLP_H

typedef struct {
    int in;                 /* 输入维度 */
    int h1;                 /* 512 */
    int h2;                 /* 256 */
    double *W1, *b1;        /* (h1 x in) 行主序, b1长度h1 */
    double *W2, *b2;        /* (h2 x h1) */
    double *W3, *b3;        /* (3 x h2) */
    double *mean, *scale;   /* 长度in */
} MLP;

/* 从目录加载权重(找W1_*x*.csv等文件), 成功返回0 */
int mlp_load(MLP *m, const char *dir);

/* x: 特征; 输出logits[3]和softmax概率prob[3] */
void mlp_forward(const MLP *m, const double *x, double logits[3], double prob[3]);

void mlp_free(MLP *m);

#endif
