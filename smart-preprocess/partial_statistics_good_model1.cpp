#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <sys/stat.h>
#ifdef WIN32
#include <direct.h>
#include <io.h>
#define READ_PATH ".\\output\\*"
#define WRITE_PATH ".\\good\\*"
#define MC1_PATH ".\\failed\\mc1\\"
#define MC2_PATH ".\\failed\\mc2\\"
#endif

#ifdef linux
#include <sys/io.h>
#include <dirent.h>
#include <unistd.h>
#define READ_PATH "./output/"
#define WRITE_PATH "./good/"
#define MC1_PATH "./failed/mc1/"
#define MC2_PATH "./failed/mc2/"
#endif

#define DISK_THRESHOLD 200000

using namespace std;

vector<vector<long>> statistics_mc1, statistics_mc2;
set<int> bad_disks_mc1, bad_disks_mc2;

void add_attributes(string filename, int type_) {
    ifstream f(filename);
    // skip the first row
    string line;
    getline(f, line);

    int count_disk = 0;
    int prev_disk = -1;

    while (getline(f, line)) {

        // diskid 
        string number;
        istringstream readstr(line);
        getline(readstr, number, ',');
        int diskid = atoi(number.c_str());
        if (prev_disk == -1) {
            count_disk++;
        } else if (prev_disk != count_disk) {
            count_disk++;
        }
        if (count_disk > DISK_THRESHOLD) {
            break;
        }

        // neglect bad drives
        if (type_ == 1 && bad_disks_mc1.count(diskid)) {
            continue;
        }
        if (type_ == 2 && bad_disks_mc2.count(diskid)) {
            continue;
        }
        
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

void read_files() {
    // mc1
    for (int piece = 0; piece < 21; piece++) {
        string filename = string(READ_PATH) + "MC1_R_" + to_string(piece) + ".csv";
        cout << "reading " << filename << '\n';
        add_attributes(filename, 1);
    }

    // mc2
    for (int piece = 0; piece < 21; piece++) {
        string filename = string(READ_PATH) + "MC2_R_" + to_string(piece) + ".csv";
        cout << "reading " << filename << '\n';
        add_attributes(filename, 2);
    }
}

void write_files() {
    cout << "writing files...\n";
    // mc1
    string file1 = string(WRITE_PATH) + "partial_statistics_mc1_model1.csv";
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
    string file2 = string(WRITE_PATH) + "partial_statistics_mc2_model1.csv";
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
}

void read_bad_lists() {
    cout << "reading bad lists...\n";
    // mc1
    string list1 = string(MC1_PATH) + "list.txt";
    ifstream f1(list1);
    if (!f1.is_open()) {
        cout << "error opening file " + list1 + "\n";
        return;
    }
    string line;
    while (getline(f1, line)) {
        bad_disks_mc1.insert(atoi(line.c_str()));
    }
    f1.close();

    // mc2
    string list2 = string(MC2_PATH) + "list.txt";
    ifstream f2(list2);
    if (!f2.is_open()) {
        cout << "error opening file " + list2 + "\n";
        return;
    }
    while (getline(f2, line)) {
        bad_disks_mc2.insert(atoi(line.c_str()));
    }
    f2.close();
}

int main() {
    read_bad_lists();
    read_files();
    write_files();
    return 0;
}