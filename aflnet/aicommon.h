#ifndef _AICOMMON_H
#define _AICOMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "klist.h"
#include "types.h"
#include "config.h"
#include "debug.h"
#include "aflnet.h"

#include <sys/file.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <stdio.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define SHM_DATA_NUM 1

#define SHM_INFO_SIZE SHM_DATA_NUM +1   //缓冲区推理结果大小，多一位用来同步
#define SHM_DATA_SIZE 4096   //缓冲区存放数据的大小

#define SHM_INFO_ID 123456789
#define SHM_INFO_DATA_ID 12345671


static u8 *python_path=NULL;
static u8 **model_args=NULL;
static u8 *weight_path=NULL;

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

static s32 shm_info_id;                    /* ID of the SHM region             */
static s32 shm_info_data_id[SHM_DATA_NUM];
static u8 *temp_out_buf[SHM_DATA_NUM];
static s32 temp_out_buf_len[SHM_DATA_NUM];

extern u32 gate_window_size;
extern u32 gate_validation_window;
extern u32 gate_topk;
extern u32 gate_validation_min_samples;
extern u32 gate_high_freq_validation_min_samples;

extern double gate_new_edge_threshold;
extern double gate_topk_ratio_threshold;
extern double gate_precision_threshold;
extern double gate_high_freq_precision_threshold;
extern double gate_validation_false_positive_threshold;
extern double gate_filter_ratio;
extern double gate_min_filter_ratio;
extern double gate_filter_ratio_step;
extern double gate_shadow_sample_ratio;
extern double gate_drift_threshold;
extern double gate_base_filter_ratio;
extern double gate_topk_high_threshold;
extern double gate_new_edge_high_threshold;
extern double gate_novel_guard_threshold;
extern double gate_high_freq_recall_threshold;
extern u32 retrain_recent_path_window;
extern u32 retrain_gain_window;
extern double retrain_precision_drop_threshold;
extern double retrain_jsd_threshold;
extern double retrain_marginal_gain_threshold;
extern double retrain_validation_sample_ratio;
extern u32 retrain_consecutive_triggers;
extern u32 retrain_cooldown_minutes;
extern double retrain_novel_false_drop_threshold;





struct ConfigInfo
{
	char key[64];
	char val[512];
};


u8 train_model();
int infer();

void remove_seeds_shm();
void setup_info_shm();


//int collect_dataset();
//void seeds_shm();

//获得文件有效行数
int getLines_ConfigFile(FILE *file);
//加载配置文件
int loadFile_ConfigFile(const char *filePath,char ***fileData,int *lines);
//解析配置文件
void parseFile_ConfigFile(char **fileData, int lines, struct ConfigInfo **info);
//获得指定配置信息
char* getInfo_ConfigFile(const char *key, struct ConfigInfo *info,int line);
//释放配置文件信息
// void destroInfo_ConfigFile(struct ConfigInfo *info);
//判断当前行是否有效
int isValid_ConfigFile(const char *buf);
//解析AI模型参数为二维数组
char** split_array(char* arr);
//解析配置文件
void parse_config();
//将kl_messages存入共享内存
void save_kl_messages_to_shm(klist_t(lms) *kl_messages);
void restart_train();
void re_collect();
int is_retrain(u64 min_wo_finds,u64 is_train_time);


#endif
