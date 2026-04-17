

#include "aicommon.h"
// int *info_bits = NULL;                /* SHM with info bitmap  */
// u8 *info_data_bit[SHM_DATA_NUM];                /* SHM with info bitmap  */


// u32 TRAIN_DATASET_NUM        = 0 ;
// u8 train_flag               = 0 ;         /* the flag of the trainning        */

//共享内存数据和通信信息
int *info_bits = NULL;                /* SHM with info bitmap  */
u8 *info_data_bit[SHM_DATA_NUM];                /* SHM with info bitmap  */


u32 TRAIN_DATASET_NUM       = 0 ;
u8 train_flag               = 0 ;   

u8 DL_mode           		= 0 ;
 
u32 infer_flag       		= 0 ,
success_infer_num           = 0 ,
success_flag                = 0 ,          /* the flag of the success          */
last_new_path               = 0 ,
train_num                   = 0 ,
cur_hash                    = 0 ,
restart_flag                = 0 ,       //重启训练
re_collect_flag             = 0 ;


u64 infer_run        		= 0 ,
infer_no_run                = 0 ,
success_time                = 0 ,
succ_infer                  = 0 ;


s32 val_pid          		= -1,                /* PID of the inference           */
rcv_pid                     = -1,
train_pid                   = -1;            /* PID of the train model           */

u32 gate_window_size                    = 256;
u32 gate_validation_window              = 64;
u32 gate_topk                           = 5;
u32 gate_validation_min_samples         = 16;
u32 gate_high_freq_validation_min_samples = 8;

double gate_new_edge_threshold              = 0.02;
double gate_topk_ratio_threshold            = 0.60;
double gate_precision_threshold             = 0.90;
double gate_high_freq_precision_threshold   = 0.85;
double gate_validation_false_positive_threshold = 0.20;
double gate_filter_ratio                    = 0.90;
double gate_min_filter_ratio                = 0.25;
double gate_filter_ratio_step               = 0.10;
double gate_shadow_sample_ratio             = 0.10;
double gate_drift_threshold                 = 0.35;
double gate_base_filter_ratio               = 0.20;
double gate_topk_high_threshold             = 0.45;
double gate_new_edge_high_threshold         = 0.05;
double gate_novel_guard_threshold           = 0.05;
double gate_high_freq_recall_threshold      = 0.60;
u32 retrain_recent_path_window              = 256;
u32 retrain_gain_window                     = 64;
double retrain_precision_drop_threshold     = 0.10;
double retrain_jsd_threshold                = 0.30;
double retrain_marginal_gain_threshold      = 0.01;
double retrain_validation_sample_ratio      = 0.10;
u32 retrain_consecutive_triggers            = 3;
u32 retrain_cooldown_minutes                = 60;
double retrain_novel_false_drop_threshold   = 0.05;

static u32 get_u32_config_or_default(const char *key, struct ConfigInfo *info,
                                     int lines, u32 default_value) {

    char *value = getInfo_ConfigFile(key, info, lines);
    if (!value || !*value) return default_value;
    return (u32)strtoul(value, NULL, 10);

}

static double get_double_config_or_default(const char *key, struct ConfigInfo *info,
                                           int lines, double default_value) {

    char *value = getInfo_ConfigFile(key, info, lines);
    if (!value || !*value) return default_value;
    return strtod(value, NULL);

}

//获得文件有效行数
int getLines_ConfigFile(FILE *file)
{

	char buf[1024] = { 0 };
	int lines = 0;
	while (fgets(buf,1024,file) != NULL)
	{
		if (!isValid_ConfigFile(buf))
		{
			continue;
		}

		memset(buf, 0, 1024);

		++lines;
	}

	//把文件指针重置到文件的开头
	fseek(file, 0, SEEK_SET);

	return lines;
}
//加载配置文件
int loadFile_ConfigFile(const char *filePath, char ***fileData, int *line)
{
	
	FILE *file = fopen(filePath, "r");
	if (NULL == file)
	{
		return 1;
	}

	int lines = getLines_ConfigFile(file);

	//给每行数据开辟内存
	char **temp = malloc(sizeof(char *) * lines);

	char buf[1024] = { 0 };

	int index = 0;

	while (fgets(buf, 1024, file) != NULL)
	{
		//如果返回false
		if (!isValid_ConfigFile(buf))
		{
			continue;
		}

		temp[index] = malloc(strlen(buf) + 1);
		strcpy(temp[index], buf);
		++index;
		//清空buf
		memset(buf, 0, 1024);
	}

	//关闭文件
	fclose(file);


	*fileData = temp;
	*line = lines;
    return 0;
}


//解析配置文件
void parseFile_ConfigFile(char **fileData, int lines, struct ConfigInfo **info)
{

	struct ConfigInfo *myinfo = malloc(sizeof(struct ConfigInfo) *lines);
	memset(myinfo, 0, sizeof(struct ConfigInfo) *lines);

	for (int i = 0; i < lines; ++i)
	{
		char *pos = strchr(fileData[i], ':');
		
		strncpy(myinfo[i].key, fileData[i], pos - fileData[i]);

		int flag = 0;
		if (fileData[i][strlen(fileData[i]) - 1] == '\n')
		{
			
			flag = 1;
		}
		
		strncpy(myinfo[i].val, pos + 1, strlen(pos + 1) - flag);
		
		// printf("key:%s val:%s\n", myinfo[i].key, myinfo[i].val);
	}

	//释放文件信息
	for (int i = 0; i < lines; ++i)
	{
		if (fileData[i]  != NULL)
		{
			free(fileData[i]);
			fileData[i] = NULL;
		}
	}


	*info = myinfo;
}
//获得指定配置信息
char* getInfo_ConfigFile(const char *key, struct ConfigInfo *info,int line)
{
	for (int i = 0; i < line; ++i)
	{
		if (strcmp(key,info[i].key) == 0)
		{
			return info[i].val;
		}
	}

	return NULL;
}
//释放配置文件信息
// void destroInfo_ConfigFile(struct ConfigInfo *info)
// {

// 	if (NULL == info)
// 	{
// 		return;
// 	}

// 	free(info);
// 	info = NULL;

// }
//判断当前行是否有效
int isValid_ConfigFile(const char *buf)
{
	if (buf[0] == '#' || buf[0] == '\n' || strchr(buf,':') == NULL)
	{
		return 0;
	}

	return 1;
}

char** split_array(char *arr ){

    // char **args_arr = NULL ;
    char **args_arr;
    char *temp ;
    int arglen = 0;
    char *pr =NULL;
    args_arr = (char**)malloc (sizeof(char**));

    temp = strtok_r( arr , " ",&pr);

    while (temp !=NULL ){
        args_arr = realloc (args_arr,(arglen + 1) * sizeof(char*)); 

        args_arr[arglen] = strdup(temp);
        temp = strtok_r(NULL , " ",&pr);
        arglen++;
    }

    args_arr = realloc (args_arr,(arglen) * sizeof(char*));
    char *terminal=NULL;
    args_arr[arglen-1]=terminal;

    return args_arr;
    // return args_arr;

}

void parse_config(){

    char **fileData = NULL;
	int lines = 0;
	struct ConfigInfo *info = NULL;
    char* temp;
    if(loadFile_ConfigFile("/tmp/ADFuzz/aflnet/config.ini", &fileData, &lines)){
        PFATAL("Cant open config.ini");
    }
    parseFile_ConfigFile(fileData, lines, &info);
    
    //解析weight路径
    weight_path = getInfo_ConfigFile("weight path",info,lines);
    //解析python路径
    python_path = getInfo_ConfigFile("python path",info,lines);
    //解析model args
    model_args  = (u8 **)split_array(getInfo_ConfigFile("model args",info,lines));
    //解析训练数据集数量
    temp = getInfo_ConfigFile("train numbers",info,lines);
    if (temp) TRAIN_DATASET_NUM = atoi(temp);

    gate_window_size = get_u32_config_or_default("gate window size", info, lines, gate_window_size);
    gate_validation_window = get_u32_config_or_default("gate validation window", info, lines, gate_validation_window);
    gate_topk = get_u32_config_or_default("gate topk", info, lines, gate_topk);
    gate_validation_min_samples = get_u32_config_or_default("gate validation min samples", info, lines, gate_validation_min_samples);
    gate_high_freq_validation_min_samples = get_u32_config_or_default("gate high freq validation min samples", info, lines, gate_high_freq_validation_min_samples);

    gate_new_edge_threshold = get_double_config_or_default("gate new edge threshold", info, lines, gate_new_edge_threshold);
    gate_topk_ratio_threshold = get_double_config_or_default("gate topk ratio threshold", info, lines, gate_topk_ratio_threshold);
    gate_precision_threshold = get_double_config_or_default("gate precision threshold", info, lines, gate_precision_threshold);
    gate_high_freq_precision_threshold = get_double_config_or_default("gate high freq precision threshold", info, lines, gate_high_freq_precision_threshold);
    gate_validation_false_positive_threshold = get_double_config_or_default("gate false positive threshold", info, lines, gate_validation_false_positive_threshold);
    gate_filter_ratio = get_double_config_or_default("gate filter ratio", info, lines, gate_filter_ratio);
    gate_min_filter_ratio = get_double_config_or_default("gate min filter ratio", info, lines, gate_min_filter_ratio);
    gate_filter_ratio_step = get_double_config_or_default("gate filter ratio step", info, lines, gate_filter_ratio_step);
    gate_shadow_sample_ratio = get_double_config_or_default("gate shadow sample ratio", info, lines, gate_shadow_sample_ratio);
    gate_drift_threshold = get_double_config_or_default("gate drift threshold", info, lines, gate_drift_threshold);
    gate_base_filter_ratio = get_double_config_or_default("gate base filter ratio", info, lines, gate_base_filter_ratio);
    gate_topk_high_threshold = get_double_config_or_default("gate topk high threshold", info, lines, gate_topk_high_threshold);
    gate_new_edge_high_threshold = get_double_config_or_default("gate new edge high threshold", info, lines, gate_new_edge_high_threshold);
    gate_novel_guard_threshold = get_double_config_or_default("gate novel guard threshold", info, lines, gate_novel_guard_threshold);
    gate_high_freq_recall_threshold = get_double_config_or_default("gate high freq recall threshold", info, lines, gate_high_freq_recall_threshold);
    retrain_recent_path_window = get_u32_config_or_default("retrain recent path window", info, lines, retrain_recent_path_window);
    retrain_gain_window = get_u32_config_or_default("retrain gain window", info, lines, retrain_gain_window);
    retrain_precision_drop_threshold = get_double_config_or_default("retrain precision drop threshold", info, lines, retrain_precision_drop_threshold);
    retrain_jsd_threshold = get_double_config_or_default("retrain jsd threshold", info, lines, retrain_jsd_threshold);
    retrain_marginal_gain_threshold = get_double_config_or_default("retrain marginal gain threshold", info, lines, retrain_marginal_gain_threshold);
    retrain_validation_sample_ratio = get_double_config_or_default("retrain validation sample ratio", info, lines, retrain_validation_sample_ratio);
    retrain_consecutive_triggers = get_u32_config_or_default("retrain consecutive triggers", info, lines, retrain_consecutive_triggers);
    retrain_cooldown_minutes = get_u32_config_or_default("retrain cooldown minutes", info, lines, retrain_cooldown_minutes);
    retrain_novel_false_drop_threshold = get_double_config_or_default("retrain novel false drop threshold", info, lines, retrain_novel_false_drop_threshold);

}


//模型训练
u8 train_model(){

    //此函数只需要开启训练模型即可

    /*所需参数：
        model_args   模型启动字符串：启动模型训练的命令       从config文件中读取
        python_path  解释器路径字符串：python执行路径        添加至环境变量
        weight_path  权重文件字符串：权重文件路径            添加至环境变量
    */


    //开启进程，子进程训练模型


    //首次训练
    if ( train_flag == 0 ){ 
        
        //保证只训练一次
        train_flag = 1;
        
        train_pid=fork();

        //失败
        if (train_pid<0) {
            PFATAL("Failed to fork train model.");
        }

        //子进程启动
        if (!train_pid){

            //分配到不同CPU核心
            static cpu_set_t mask_child;
            CPU_ZERO(&mask_child);
            CPU_SET(2, &mask_child);

            if (sched_setaffinity(0, sizeof(mask_child), &mask_child) == -1) {
                PFATAL("Failed to bind child'cpu.");
            }

            execv((char *)python_path, (char *const *)model_args);



            return -1;    //训练模型失败
        }

    }

    else if ( train_flag > 1 && !re_collect_flag){   //每次执行一次
        re_collect_flag = 1;    
        re_collect();   //通知一次AI模型收集完成
    }
    
 

    //判断weight文件是否存在，存在则训练完成，开启预测模式
    //char * model_path ="/home/yu/公共的/FoRTE-FuzzBench/binutils/output/ganomaly/testMain/domain.txt";

    if ( (access(weight_path, F_OK)) == 0 ) {

        return 1;   //说明weight文件存在

    }
            

    

    return 0;  //正常训练但是还没训练完成

    //第二次训练

}



//模型推理
int infer(){

    //推理


    if (infer_flag == 0) {

        pthread_mutex_lock(&mutex);

        info_bits[SHM_DATA_NUM] = 1;

        pthread_mutex_unlock(&mutex);

        infer_flag = 1;
    }


    if(info_bits[SHM_DATA_NUM] == 0) { //说明model.validata的结果已经出来
        infer_flag=0;
        return 1;
    }

    return 0;


}



//清除种子缓冲区
void remove_seeds_shm(){


    shmctl(shm_info_id, IPC_RMID, NULL);
    int i;
    for (i = 0; i < SHM_DATA_NUM; i++) {
        shmctl(shm_info_data_id[i], IPC_RMID, NULL);
    }

}

//设置消息缓冲区通信
void setup_info_shm(){
    
    shm_info_id = shmget(SHM_INFO_ID, SHM_INFO_SIZE, IPC_CREAT | IPC_EXCL | 0644);

    if (shm_info_id < 0) PFATAL("shmget() failed");

    atexit(remove_seeds_shm);


    info_bits = shmat(shm_info_id, NULL, 0);

    if (info_bits == (void *) -1) PFATAL("shmat() failed");


    int i;

    for (i = 0;i < SHM_DATA_NUM;i++) {

        shm_info_data_id[i] = shmget(SHM_INFO_DATA_ID+i, SHM_DATA_SIZE, IPC_CREAT | IPC_EXCL | 0644);

        if (shm_info_data_id[i] < 0) PFATAL("shmget() failed");

        atexit(remove_seeds_shm);


        info_data_bit[i] = shmat(shm_info_data_id[i], NULL, 0);

        if (info_data_bit[i] == (void *) -1) PFATAL("shmat() failed");

    }

}


void restart_train(){


    pthread_mutex_lock(&mutex);

    info_bits[SHM_DATA_NUM] = 2; //通知模型等待fuzzer收集种子完成

    pthread_mutex_unlock(&mutex);

    re_collect_flag = 0;        //打开收集

    success_infer_num++;      //开启模型训练次数统计

    //删除之前的数据文件


}

void re_collect(){


    pthread_mutex_lock(&mutex);

    info_bits[SHM_DATA_NUM] = 3; //通知模型重新训练

    pthread_mutex_unlock(&mutex);

    restart_flag = 0 ;

}

int is_retrain(u64 min_wo_finds,u64 is_train_time){
    /*restart mode to train*/

    if (is_train_time > 120 && min_wo_finds > 60 ){
        return 1 ;
    }

    return 0;
}
