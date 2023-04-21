#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>
#include <stdio.h>
#include <sys/stat.h>

#ifdef WIN32
#include <direct.h>
#include <io.h>
#define PATH ".\\*"
#define OUTPUT_PATH ".\\output\\"
#endif

#ifdef linux
#include <sys/io.h>
#include <dirent.h>
#include <unistd.h>
#define PATH "./"
#define OUTPUT_PATH "./output/"
#endif

using namespace std;


enum attribute {
    READ_ERROR_RATE=1,                          // C 列
    THROUGHPUT_PERFORMANCE=2,
    REALLOCATED_SECTORS_COUNT=5,                // E 列
    READ_CHANNEL_MARGIN=6,
    SEEK_SECTOR_RATE=7,
    SEEK_TIME_PERFORMANCE=8,
    POWER_ON_HOURS=9,                           
    CALIBRATION_RETRY_COUNT=11,
    POWER_CYCLE_COUNT=12,                       
    SOFT_READ_ERROR_RATE=13,
    AVAILABLE_RESERVED_SPACE=170,              
    SSD_PROGRAM_FAIL_COUNT=171,                 
    SSD_ERASE_FAIL_COUNT=172,                   
    SSD_WEAR_LEVELING_COUNT=173,               
    UNEXPECTED_POWER_LOSS_COUNT=174,            
    WEAR_RANGE_DELTA=177,                      
    UNUSED_RESERVED_BLOCK_COUNT_TOTAL=180,      
    PROGRAM_FAIL_COUNT_TOTAL=181,              
    ERASE_FAIL_COUNT=182,                   
    SATA_DOWNSHIFT_ERROR_COUNT=183,            
    END_TO_END_ERROR=184,                       
    REPORTED_UNCORRECTABLE_ERRORS=187,         
    COMMAND_TIMEOUT=188,                       
    TEMPERATURE_DIFFERENCE=190,
    G_SENDSE_ERROR_RATE=191,                
    UNSAFE_SHUTDOWN_COUNT=192,
    LOAD_CYCLE_COUNT=193,
    TEMPERATURE=194,                           
    HARDWARE_ECC_RECOVERED=195,                
    REALLOCATION_EVENT_COUNT=196,              
    CURRENT_PENDING_SECTOR_COUNT=197,           
    OFFLINE_UNRECORRECTABLE_SECTOR_COUNT=198,  
    ULTRADMA_CRC_ERROR_COUNT=199,             
    MULTI_ZONE_ERROR_RATE=200,
    SOFT_ECC_CORRECTION=204,                   
    THERMAL_ASPERITY_RATE=205,
    VIBRATION_DURING_WRITE=211,                 
    MEDIA_WEAROUT_INDICATOR=233,               
    TOTAL_LBAS_WRITTEN=241,
    TOTAL_LBAS_READ=242,
    TOTAL_LBAS_READ_EXPANDED=244,
    POWER_LOSS_PROTECTION_FAILURE=175,          
    ENDURANCE_REMAINING=233                     
};

vector<int> attribute_nums = {1,2,5,6,7,8,9,11,12,13,170,173,174,177,180,181,182,183,184,187,188,190,191,192,193,
194,195,196,197,198,199,200,204,205,211,233,241,242,245,233};

vector<vector<vector<long>>> disk_saves_1r, disk_saves_1n, disk_saves_2r, disk_saves_2n;

int traversefiles(string path, vector <string>& files,int file_number, vector <string>& originalFileNmae){
    //int file_num=0;

    //file directory
    
    string inPath = path;//scan all file  
    
    #ifdef WIN32
    //find handle
    long handle;
    struct _finddata_t fileinfo;
    string p;
    //find handle in first round
    handle = _findfirst(inPath.c_str(),&fileinfo);
    if(handle == -1)
        return -1;
    do
    {   
        
        //print the name of the file that has been found
        if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))  
            {  
                //cout<<"in func fileinfo.name is: "<<fileinfo.name<<endl;
                originalFileNmae.push_back(fileinfo.name);

                files.push_back(p.assign(path).erase((p.assign(path)).length()-1,1).append(fileinfo.name));
                //cout<<"file path: "<<p.assign(path).erase((p.assign(path)).length()-1,1).append(fileinfo.name)<<endl;
                file_number++;
            }
    } while (!_findnext(handle,&fileinfo));
    //cout<<"file_nume is: "<<file_num<<endl;
    _findclose(handle);

    return file_number;
    #endif

    #ifdef linux
    DIR *dir;
    if ((dir=opendir(path.c_str())) == NULL) {
        {
            string errMsg = "error opening dir " + path;
            perror(errMsg.c_str());
            exit(1);
        }
    }
    
    struct dirent *ptr;
    string p;
    while ((ptr=readdir(dir)) != NULL) {
        if ((strcmp(ptr->d_name, ".") == 0) || strcmp(ptr->d_name, "..") == 0) {
            // current directory or parent directory
            continue;
        } else {
            originalFileNmae.push_back(ptr->d_name);

            files.push_back(p.assign(path).append(ptr->d_name));
            file_number++;
        }
    }
    closedir(dir);
    return file_number;
    #endif

}

int time2index(int time) {
    int index = -1;
    if (time > 20181200) {
        index = time - 20181201 + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30;
    } else if (time > 20181100) {
        index = time - 20181101 + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31;
    } else if (time > 20181000) {
        index = time - 20181001 + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30;
    } else if (time > 20180900) {
        index = time - 20180901 + 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31;
    } else if (time > 20180800) {
        index = time - 20180801 + 31 + 28 + 31 + 30 + 31 + 30 + 31;
    } else if (time > 20180700) {
        index = time - 20180701 + 31 + 28 + 31 + 30 + 31 + 30;
    } else if (time > 20180600) {
        index = time - 20180601 + 31 + 28 + 31 + 30 + 31;
    } else if (time > 20180500) {
        index = time - 20180501 + 31 + 28 + 31 + 30;
    } else if (time > 20180400) {
        index = time - 20180401 + 31 + 28 + 31;
    } else if (time > 20180300) {
        index = time - 20180301 + 31 + 28;
    } else if (time > 20180200) {
        index = time - 20180201 + 31;
    } else {
        index = time - 20180101;
    }
    return index;
}

/*
* function: parse file
*
*/

void parse(string in_file, int piece_id) {
    ifstream f(in_file);
    if (!f.is_open()) {
        cout << "error opening file\n";
        return;
    }
    // cout << "in_file is " << in_file << endl;
    
    // skip the first row
    string line;
    getline(f, line);

    int disk_min = piece_id * 10000;
    int disk_max = (piece_id+1) * 10000;

    // return;

    while (getline(f, line)) {
        istringstream readstr(line);
        string number;

        // disk id
        getline(readstr, number, ',');
        unsigned int disk_id = atoi(number.c_str());
        // cout << "disk id is " << disk_id << endl;
        if (disk_id < disk_min || disk_id >= disk_max) {
            continue;
        }

        int disk_index = disk_id - disk_min;
        // cout << "disk_index: " << disk_index << endl;

        // timestamp (ds)
        getline(readstr, number, ',');
        int ds = atoi(number.c_str());
        int time_index = 0;
        if (ds > 20190000) {
            time_index = 365;
            ds -= 10000;
        }
        time_index += time2index(ds);
        // cout << "ds=" << ds << ", time_index=" << time_index << '\n';

        // cout << "time_index: " << time_index << endl;

        // model
        getline(readstr, number, ',');
        string model = number;
        if (model != "MC1" && model != "MC2") {
            if (model != "MA1" && model != "MA2" && model != "MB1" && model != "MB2") {
                cout << "error! model=" << model << '\n';
            }
            continue;
        } else if (model == "MC1") {
            for (int i = 0; i < 43; i++) {
                // normalized
                getline(readstr, number, ',');
                disk_saves_1n[disk_index][time_index][i] = (number != "") ? atol(number.c_str()) : -1;
                // raw
                getline(readstr, number, ',');
                disk_saves_1r[disk_index][time_index][i] = (number != "") ? atol(number.c_str()) : -1;
                if (i == 1) {
                    // neglect n3r3, n4r4
                    for (int j = 0; j < 4; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 6) {
                    // neglect n10r10
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 22) {
                    // neglect n189r189
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 35) {
                    // neglect n206r206, n207r207
                    for (int j = 0; j < 4; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 41) {
                    // neglect n245r245
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                }
            }
        } else if (model == "MC2") {
            for (int i = 0; i < 43; i++) {
                // normalized
                getline(readstr, number, ',');
                disk_saves_2n[disk_index][time_index][i] = (number != "") ? atol(number.c_str()) : -1;
                // raw
                getline(readstr, number, ',');
                disk_saves_2r[disk_index][time_index][i] = (number != "") ? atol(number.c_str()) : -1;
                if (i == 1) {
                    // neglect n3r3, n4r4
                    for (int j = 0; j < 4; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 6) {
                    // neglect n10r10
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 22) {
                    // neglect n189r189
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 35) {
                    // neglect n206r206, n207r207
                    for (int j = 0; j < 4; j++) {
                        getline(readstr, number, ',');
                    }
                } else if (i == 41) {
                    // neglect n245r245
                    for (int j = 0; j < 2; j++) {
                        getline(readstr, number, ',');
                    }
                }
            }
        }
        
    } 
}

void traverse(int piece_id) {
    struct stat filetype;
    int Rfolder_num = 0;
    string eventlog_name = "smartlog";
    int file_num = 0;
    int Rfile_num = 0;
    string binFilePath;

    std::string temp;
	vector <string> folders;
    vector <string> name_tmp;

    string filePath = PATH; 
    string dir_name_temp;
    int pedata_num = 0;
    int eventlog_num = 0;

    

    file_num =  traversefiles(filePath, folders,file_num,name_tmp);
	for (int i = 0; i < folders.size(); i++) {
        vector <string> binFiles = vector <string>();
        vector<string> output_name = vector <string>();
        #ifdef WIN32
        if (stat(folders[i].c_str(),&filetype)==0){
            if(filetype.st_mode & S_IFDIR){
        #endif
        #ifdef linux
        struct stat s_buf;
        stat(folders[i].c_str(), &s_buf);
        if (S_ISDIR(s_buf.st_mode)) {
            if (true) {
        #endif
                Rfolder_num ++;

                string::size_type eventidx = folders[i].find(eventlog_name);
                if(eventidx != string::npos)
                {
                    #ifdef WIN32
                    binFilePath = folders[i] + "\\*";
                    #endif
                    #ifdef linux
                    binFilePath = folders[i] + "/";
                    #endif
                    traversefiles(binFilePath, binFiles, Rfile_num,output_name);
                    
                    
                    for(int j=0;j<binFiles.size();j++){
                            #ifdef WIN32
                            string tempPath = folders[i]+"\\"+output_name[j];
                            #endif

                            #ifdef linux
                            string tempPath = folders[i]+"/"+output_name[j];
                            #endif
                            parse(tempPath, piece_id);
                    }
                }
            }
            else if (filetype.st_mode & S_IFREG){
                Rfile_num ++;
            }
        }

	}

    string out_path = string(OUTPUT_PATH);

    // normalized, model MC1
    ofstream of18mc1n;
    of18mc1n.open(out_path+"MC1_N_"+to_string(piece_id)+".csv", ios::out | ios::trunc);

    // normalized, model MC2
    ofstream of18mc2n;
    of18mc2n.open(out_path+"MC2_N_"+to_string(piece_id)+".csv", ios::out | ios::trunc);

    of18mc1n << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232\n";
    of18mc2n << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232\n";
    
    int disk_id = piece_id * 10000;
    for (auto disk_save: disk_saves_1n) {
        // 10000 disk_save
        int day = 1;
        for (auto time_info: disk_save) {
            // 365*2 time_info
            of18mc1n << disk_id << ",";
            of18mc1n << day++ << ",";
            for (auto attribute: time_info) {
                // 43 attributes
                of18mc1n << attribute << ",";
            }
            of18mc1n << '\n';
        }
        disk_id++;
    }

    disk_id = piece_id * 10000;
    for (auto disk_save: disk_saves_2n) {
        // 10000 disk_save
        int day = 1;
        for (auto time_info: disk_save) {
            // 365*2 time_info
            of18mc2n << disk_id << ",";
            of18mc2n << day++ << ",";
            for (auto attribute: time_info) {
                // 43 attributes
                of18mc2n << attribute << ",";
            }
            of18mc2n << '\n';
        }
        disk_id++;
    }
    
    of18mc1n.close();
    of18mc2n.close();

    // 2018, raw, model MC1
    ofstream of18mc1r;
    of18mc1r.open(out_path+"MC1_R_"+to_string(piece_id)+".csv", ios::out | ios::trunc);

    // 2018, raw, model MC2
    ofstream of18mc2r;
    of18mc2r.open(out_path+"MC2_R_"+to_string(piece_id)+".csv", ios::out | ios::trunc);

    of18mc1r << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232\n";
    of18mc2r << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232\n";
    
    disk_id = piece_id * 10000;
    for (auto disk_save: disk_saves_1r) {
        // 10000 disk_save
        int day = 1;
        for (auto time_info: disk_save) {
            // 365*2 time_info
            of18mc1r << disk_id << ",";
            of18mc1r << day++ << ",";
            for (auto attribute: time_info) {
                // 43 attributes
                of18mc1r << attribute << ",";
            }
            of18mc1r << '\n';
        }
        disk_id++;
    }

    disk_id = piece_id * 10000;
    for (auto disk_save: disk_saves_2r) {
        // 10000 disk_save
        int day = 1;
        for (auto time_info: disk_save) {
            // 365*2 time_info
            of18mc2r << disk_id << ",";
            of18mc2r << day++ << ",";
            for (auto attribute: time_info) {
                // 43 attributes
                of18mc2r << attribute << ",";
            }
            of18mc2r << '\n';
        }
        disk_id++;
    }

    of18mc1r.close();
    of18mc2r.close();
}

int main() {

    for (int piece_id = 0; piece_id < 21; piece_id++) {
        // initialize
        disk_saves_1n = vector<vector<vector<long>>>();
        disk_saves_1n.resize(10000);
        for (int i = 0; i < 10000; i++) {
            disk_saves_1n[i].resize(365*2);
            for (int j = 0; j < 365*2; j++) {
                disk_saves_1n[i][j].resize(43, -1);
            }
        }

        disk_saves_1r = vector<vector<vector<long>>>();
        disk_saves_1r.resize(10000);
        for (int i = 0; i < 10000; i++) {
            disk_saves_1r[i].resize(365*2);
            for (int j = 0; j < 365*2; j++) {
                disk_saves_1r[i][j].resize(43, -1);
            }
        }

        disk_saves_2n = vector<vector<vector<long>>>();
        disk_saves_2n.resize(10000);
        for (int i = 0; i < 10000; i++) {
            disk_saves_2n[i].resize(365*2);
            for (int j = 0; j < 365*2; j++) {
                disk_saves_2n[i][j].resize(43, -1);
            }
        }

        disk_saves_2r = vector<vector<vector<long>>>();
        disk_saves_2r.resize(10000);
        for (int i = 0; i < 10000; i++) {
            disk_saves_2r[i].resize(365*2);
            for (int j = 0; j < 365*2; j++) {
                disk_saves_2r[i][j].resize(43, -1);
            }
        }

        // traverse
        traverse(piece_id);
    }
    
    return 0;
}