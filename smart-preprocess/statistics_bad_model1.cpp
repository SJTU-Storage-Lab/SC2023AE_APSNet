/*
THIS IS MODEL 1

model 1:
temperature                 194
hours                       9
wear leveling               173
program errors              171
erase errors                172
(optional) read error rate  1

model 2:
temperature                 194
hours                       9
command timeout             188
unused reserved blocks      180
unexpected power loss       174
sata errors                 183
(optional) read error rate  1
uncorrectable errors        187
*/

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
#define MC1_PATH ".\\failed\\mc1\\"
#define MC2_PATH ".\\failed\\mc2\\"
#endif

#ifdef linux
#include <sys/io.h>
#include <dirent.h>
#include <unistd.h>
#define PATH "./"
#define MC1_PATH "./failed/mc1/"
#define MC2_PATH "./failed/mc2/"
#endif

using namespace std;

vector<vector<long>> statistics_mc1, statistics_mc2;

void read_file(string filename, int type_) {
    ifstream f(filename);
    if (!f.is_open()) {
        cout << "error opening file " + filename + "\n";
        return;
    }

    // skip the first row
    string line;
    getline(f, line);

    while (getline(f, line)) {
    
        // diskid 
        string number;
        istringstream readstr(line);
        getline(readstr, number, ',');
        int diskid = atoi(number.c_str());
        
        // read error rate - 1
        for (int i = 0; i < 2; i++) {
            getline(readstr, number, ',');
        }
        long read_error_rate = atol(number.c_str());

        // hours - 9
        for (int i = 0; i < 2; i++) {
            getline(readstr, number, ',');
        }
        long hours = atol(number.c_str());

        // program error counts - 171
        for (int i = 0; i < 3; i++) {
            getline(readstr, number, ',');
        }
        long program_errors = atol(number.c_str());

        // erase error counts - 172
        getline(readstr, number, ',');
        long erase_errors = atol(number.c_str());

        // wear leveling - 173
        getline(readstr, number, ',');
        long wear = atol(number.c_str());

        // temperature - 194
        for (int i = 0; i < 7; i++) {
            getline(readstr, number, ',');
        }
        long T = atol(number.c_str());
        
        if (type_ == 1) {
            statistics_mc1.push_back({diskid, T, hours, wear, program_errors, erase_errors, read_error_rate});
        } else if (type_ == 2) {
            statistics_mc2.push_back({diskid, T, hours, wear, program_errors, erase_errors, read_error_rate});
        }
    }

    f.close();
}

void write_files() {
    // // mc1
    string file1 = string(MC1_PATH) + "statistics_model1.csv";
    cout << "writing " << file1 << '\n';
    ofstream of1;
    of1.open(file1, ios::out | ios::trunc);

    of1 << "diskid,T (194),hours (9),wear leveling (173),program errors (171),erase errors (172),read error rate (1),\n";
    for (auto r: statistics_mc1) {
        for (auto attribute: r) {
            of1 << attribute << ',';
        }
        of1 << '\n';
    }
    of1.close();

    // mc2
    string file2 = string(MC2_PATH) + "statistics_model1.csv";
    cout << "writing " << file2 << '\n';
    ofstream of2;
    of2.open(file2, ios::out | ios::trunc);

    of2 << "diskid,T (194),hours (9),wear leveling (173),program errors (171),erase errors (172),read error rate (1),\n";
    for (auto r: statistics_mc2) {
        for (auto attribute: r) {
            of2 << attribute << ',';
        }
        of2 << '\n';
    }
    of2.close();
    cout << "end of writing files\n";
}

void read_files() {
    // mc1
    string list1 = string(MC1_PATH) + "list.txt";
    ifstream f1(list1);
    if (!f1.is_open()) {
        cout << "error opening file " + list1 + "\n";
        return;
    }
    vector<int> disks;
    string line;
    while (getline(f1, line)) {
        disks.push_back(atoi(line.c_str()));
    }
    f1.close();

    for (auto disk: disks) {
        string filename = string(MC1_PATH) + to_string(disk) + ".csv";
        read_file(filename, 1);
    }

    // mc2
    string list2 = string(MC2_PATH) + "list.txt";
    ifstream f2(list2);
    if (!f2.is_open()) {
        cout << "error opening file " + list2 + "\n";
        return;
    }
    disks = vector<int>();
    while (getline(f2, line)) {
        disks.push_back(atoi(line.c_str()));
    }
    f2.close();

    for (auto disk: disks) {
        string filename = string(MC2_PATH) + to_string(disk) + ".csv";
        read_file(filename, 2);
    }
}

int main() {
    read_files();
    write_files();
    return 0;
}