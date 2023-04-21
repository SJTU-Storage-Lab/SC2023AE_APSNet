#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <stdlib.h>

#ifdef WIN32
#include <direct.h>
#include <io.h>
#define PATH ".\\output\\"
#define LABEL_PATH ".\\ssd_failure_label.csv"
#define MC1_PATH ".\\failed\\failed_mc1.csv"
#define MC2_PATH ".\\failed\\failed_mc2.csv"
#endif

#ifdef linux
#include <sys/io.h>
#include <dirent.h>
#include <unistd.h>
#define PATH "./output/"
#define LABEL_PATH "./ssd_failure_label.csv"
#define MC1_PATH "./failed/failed_mc1.csv"
#define MC2_PATH "./failed/failed_mc2.csv"
#endif

using namespace std;

vector<int> time_v = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
map<int, int> diskid2index1, diskid2index2;

vector<vector<long>> mc1_disks, mc2_disks;

int time_fail(string time) {
    int failure_time = 0;

    int pos = time.find('/');
    int year = atoi(time.substr(0, pos).c_str());
    time = time.substr(pos+1, time.length()-5);
    if (year == 2019) {
        failure_time += 365;
    }

    pos = time.find('/');
    int month = atoi(time.substr(0, pos).c_str());
    failure_time += time_v[month-1];
    time = time.substr(pos+1, time.length()-1-time.substr(0, pos).length());

    pos = time.find(' ');
    int date = atoi(time.substr(0, pos).c_str());
    failure_time += date;
    // cout << "time: " << time << ", year: " << year << ", month: " << month << ", date: " << date << endl;

    return failure_time;
}

vector<pair<int, int>> mc1, mc2;
map<int, vector<pair<int, int>>> mc1_map, mc2_map;

void read_bad_labels() {
    cout << "reading bad labels...\n";

    ifstream f(LABEL_PATH);
    if (!f.is_open()) {
        cout << "error opening file " + string(LABEL_PATH) + "\n";
        return;
    }

    // skip the first row
    string line;
    getline(f, line);

    while (getline(f, line)) {

        istringstream readstr(line);
        string number;

        // model
        getline(readstr, number, ',');
        string model = number;
        if (model != "MC1" && model != "MC2") {
            continue;
        }

        // failure_time
        getline(readstr, number, ',');
        int failure_time = time_fail(number);

        // disk_id
        getline(readstr, number, ',');
        int disk_id = atoi(number.c_str());

        if (model == "MC1") {
            mc1.push_back(make_pair(disk_id, failure_time));
        } else if (model == "MC2") {
            mc2.push_back(make_pair(disk_id, failure_time));
        }
    }

    cout << mc1.size() << endl;
    cout << mc2.size() << endl;

    for (int i = 0; i < mc1.size(); i++) {
        // diskid, time
        diskid2index1[mc1[i].first] = i;
    }

    for (int i = 0; i < mc2.size(); i++) {
        // diskid, time
        diskid2index2[mc2[i].first] = i;
    }

    sort(mc1.begin(), mc1.end());
    sort(mc2.begin(), mc2.end());

    for (int i = 0; i < mc1.size(); i++) {
        int piece_id = mc1[i].first / 10000;
        mc1_map[piece_id].push_back(mc1[i]);
    }

    cout << "mc1_map: \n";
    for (int i = 0; i < mc1_map.size(); i++) {
        cout << i << ", " << mc1_map[i].size() << endl;
    }

    for (int i = 0; i < mc2.size(); i++) {
        int piece_id = mc2[i].first / 10000;
        mc2_map[piece_id].push_back(mc2[i]);
    }

    cout << "mc2_map: \n";
    for (int i = 0; i < mc2_map.size(); i++) {
        cout << i << ", " << mc2_map[i].size() << endl;
    }
}

void write_file(string out_mc1, string out_mc2) {
    cout << "writing files...\n";

    ofstream of1;
    of1.open(out_mc1, ios::out | ios::trunc);
    
    of1 << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232,\n";

    for (auto v: mc1_disks) {
        // v represents a row
        for (auto attribute: v) {
            of1 << attribute << ',';
        }
        of1 << '\n';
    }

    of1.close();

    ofstream of2;
    of2.open(out_mc2, ios::out | ios::trunc);
    
    of2 << "disk_id,ds,1,2,5,6,7,8,9,11,12,13,170,171,172,173,174,177,180,181,182,183,184,187,188,190,191,192,193,194,195,196,197,198,199,200,204,205,211,233,241,242,244,175,232,\n";

    for (auto v: mc2_disks) {
        // v represents a row
        for (auto attribute: v) {
            of2 << attribute << ',';
        }
        of2 << '\n';
    }

    of2.close();
}

void read_data(string path) {
    cout << "reading data...\n";
    // read data into mc1_disks and mc2_disks

    // mc1
    cout << mc1_map.size() << endl;
    for (int piece = 0; piece < mc1_map.size(); piece++) {
        string filename = path + "MC1_R_" + to_string(piece) + ".csv";
        ifstream f;
        f.open(filename);
        cout << "opening file " << filename << endl;

        string line;
        // skip the first row
        getline(f, line);

        auto v = mc1_map[piece];
        int find_disk = v[0].first;
        int cur_index = 0;

        while (getline(f, line)) {
            string number;
            istringstream readstr(line);

            getline(readstr, number, ',');
            int diskid = atoi(number.c_str());

            if (diskid < find_disk) {
                continue;
            } else if (diskid > find_disk) {
                if (cur_index < v.size() - 1) {
                    cur_index++;
                    find_disk = v[cur_index].first;
                } else {
                    // end of traversing this file
                    break;
                }
            }
            if (diskid == find_disk) {
                vector<long> attributes;
                attributes.push_back(diskid);
                while (getline(readstr, number, ',')) {
                    attributes.push_back(atol(number.c_str()));
                }
                mc1_disks.push_back(attributes);
            }
        }

        f.close();
    }

    // mc2

    for (int piece = 0; piece < mc2_map.size(); piece++) {
        string filename = path + "MC2_R_" + to_string(piece) + ".csv";
        ifstream f;
        f.open(filename);
        cout << "opening file " << filename << endl;

        string line;
        // skip the first row
        getline(f, line);

        auto v = mc2_map[piece];
        int find_disk = v[0].first;
        int cur_index = 0;

        while (getline(f, line)) {
            string number;
            istringstream readstr(line);

            getline(readstr, number, ',');
            int diskid = atoi(number.c_str());

            if (diskid < find_disk) {
                continue;
            } else if (diskid > find_disk) {
                if (cur_index < v.size() - 1) {
                    cur_index++;
                    find_disk = v[cur_index].first;
                } else {
                    // end of traversing this file
                    break;
                }
            }
            if (diskid == find_disk) {
                vector<long> attributes;
                attributes.push_back(diskid);
                while (getline(readstr, number, ',')) {
                    attributes.push_back(atol(number.c_str()));
                }
                mc2_disks.push_back(attributes);
            }
        }

        f.close();
    }

    
}

int main() {
    read_bad_labels();
    read_data(PATH);
    write_file(MC1_PATH, MC2_PATH);
    return 0;
}