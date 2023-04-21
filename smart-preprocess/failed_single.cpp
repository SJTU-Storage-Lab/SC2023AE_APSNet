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
#define MC1_FILE ".\\failed\\failed_mc1.csv"
#define MC2_FILE ".\\failed\\failed_mc2.csv"
#define MC1_PATH ".\\failed\\mc1\\"
#define MC2_PATH ".\\failed\\mc2\\"
#endif

#ifdef linux
#include <sys/io.h>
#include <dirent.h>
#include <unistd.h>
#define PATH "./output/"
#define LABEL_PATH "./ssd_failure_label.csv"
#define MC1_FILE "./failed/failed_mc1.csv"
#define MC2_FILE "./failed/failed_mc2.csv"
#define MC1_PATH "./failed/mc1/"
#define MC2_PATH "./failed/mc2/"
#endif

using namespace std;

vector<vector<long>> mc1_disks, mc2_disks;

void process_files() {
    cout << "reading files...\n";

    // mc1
    ifstream f(MC1_FILE);
    if (!f.is_open()) {
        cout << "error opening file " << MC1_FILE << "\n";
        return;
    } else {
        cout << MC1_FILE << " successfully opens.\n";
    }

    // skip the first row
    string line;
    getline(f, line);

    vector<vector<long>> attributes;
    int last_disk_id = -1;

    while (getline(f, line)) {
        istringstream readstr(line);
        string number;

        // diskid
        getline(readstr, number, ',');
        int diskid = atoi(number.c_str());
        vector<long> row;
        row.push_back(diskid);

        while (getline(readstr, number, ',')) {
            row.push_back(atol(number.c_str()));
        }

        if (last_disk_id != diskid && attributes.size() > 0) {
            ofstream of;
            string filename = string(MC1_PATH) + to_string(last_disk_id) + ".csv";
            of.open(filename, ios::out | ios::trunc);

            of << "disk_id,ds,1,5,9,12,170,171,172,173,174,180,183,184,187,188,194,195,196,197,198,199,211\n";

            for (int i = 0; i < attributes.size(); i++) {
                for (int j = 0; j < attributes[i].size(); j++) {
                    of << attributes[i][j] << ',';
                }
                of << '\n';
            }

            of.close();

            attributes = vector<vector<long>>();
        }

        attributes.push_back(row);
        last_disk_id = diskid;
        
    }

    ofstream of1;
    string filename = string(MC1_PATH) + to_string(last_disk_id) + ".csv";
    of1.open(filename, ios::out | ios::trunc);

    of1 << "disk_id,ds,1,5,9,12,170,171,172,173,174,180,183,184,187,188,194,195,196,197,198,199,211\n";

    for (int i = 0; i < attributes.size(); i++) {
        for (int j = 0; j < attributes[i].size(); j++) {
            of1 << attributes[i][j] << ',';
        }
        of1 << '\n';
    }

    of1.close();

    f.close();

    // mc2
    ifstream f2(MC2_FILE);
    if (!f2.is_open()) {
        cout << "error opening file " << MC2_FILE << "\n";
        return;
    } else {
        cout << MC2_FILE << " successfully opens.\n";
    }

    // skip the first row
    getline(f2, line);

    attributes = vector<vector<long>>();
    last_disk_id = -1;

    while (getline(f2, line)) {
        istringstream readstr(line);
        string number;

        // diskid
        getline(readstr, number, ',');
        int diskid = atoi(number.c_str());
        vector<long> row;
        row.push_back(diskid);

        while (getline(readstr, number, ',')) {
            row.push_back(atol(number.c_str()));
        }

        if (last_disk_id != diskid && attributes.size() > 0) {
            ofstream of;
            string filename = string(MC2_PATH) + to_string(last_disk_id) + ".csv";
            of.open(filename, ios::out | ios::trunc);

            of << "disk_id,ds,1,5,9,12,170,171,172,173,174,180,183,184,187,188,194,195,196,197,198,199,211\n";

            for (int i = 0; i < attributes.size(); i++) {
                for (int j = 0; j < attributes[i].size(); j++) {
                    of << attributes[i][j] << ',';
                }
                of << '\n';
            }

            of.close();

            attributes = vector<vector<long>>();
        }

        attributes.push_back(row);
        last_disk_id = diskid;
        
    }

    ofstream of2;
    filename = string(MC2_PATH) + to_string(last_disk_id) + ".csv";
    of2.open(filename, ios::out | ios::trunc);

    of2 << "disk_id,ds,1,5,9,12,170,171,172,173,174,180,183,184,187,188,194,195,196,197,198,199,211\n";

    for (int i = 0; i < attributes.size(); i++) {
        for (int j = 0; j < attributes[i].size(); j++) {
            of2 << attributes[i][j] << ',';
        }
        of2 << '\n';
    }

    of2.close();

    f2.close();
}

int main() {
    process_files();
    return 0;
}