#include <fstream>
#include <iostream>
#include <random>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <sstream>

#include "rank_bm25.hpp"

template <class T>
void print_vec(const std::vector<T>& vec) {
    std::cout << '[';
    for (auto it=vec.begin(); it != vec.end() - 1; ++it) {
        std::cout << *it << ", ";
    }
    std::cout << vec.back() << "]\n";
}

void test_scoring() {
    std::vector<std::vector<std::string>> corpus = {
        {"Hello","there","good","man!"},
        {"It","is","quite","windy","in","London"},
        {"How","is","the","weather","today?"}
    };
    std::vector<std::string> query = {"there", "is", "London"};
    BM25Okapi bm25(corpus);
    std::vector<double> scores = bm25.get_scores(query);
    print_vec(scores); // [0.561347, 0.569072, 0.109463]
    BM25L bm25l(corpus);
    scores = bm25l.get_scores(query);
    print_vec(scores); // [2.20092, 2.34413, 1.81354]
    BM25Plus bm25p(corpus);
    scores = bm25p.get_scores(query);
    print_vec(scores); // [4.98914, 5.37348, 4.15888]
    query = {"the", "man"};
    print_vec(bm25.get_top_n(query));
    print_vec(bm25l.get_top_n(query));
    print_vec(bm25p.get_top_n(query));
}

std::vector<std::vector<std::string>> generate_random_corpus() {
    std::ifstream file("/usr/share/dict/words");
    std::vector<std::string> words;
    std::string word;
    while (file >> word) {
        if (word.size() < 8) {
            words.push_back(word);
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(words.begin(), words.end(), g);
    words.resize(1000);

    std::uniform_int_distribution<> dis(3, 10);

    std::vector<std::vector<std::string>> corpus;
    for (int i = 0; i < 10000; ++i) {
        int len = dis(g);
        std::vector<std::string> doc;
        for (int j = 0; j < len; ++j) {
            doc.push_back(words[dis(g) % 1000]);
        }
        corpus.push_back(doc);
    }

    return corpus;
}

std::vector<std::string> split(const std::string& str, char delimiter=' ') {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream ss(str);
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

#ifdef __APPLE__
void print_memory_usage() {
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(),
                                   TASK_BASIC_INFO,
                                   (task_info_t)&info,
                                   &size);
    if (kerr == KERN_SUCCESS) {
        std::cout << "Memory usage: " << info.resident_size / 1024 << " KB\n";
    } else {
        std::cerr << "Failed to get task info: " << kerr << "\n";
    }
}
#endif

void repl() {
    auto corpus = generate_random_corpus();
    std::cout << "Corpus size: " << corpus.size() << std::endl;
    double avg_len = std::accumulate(corpus.begin(), corpus.end(), 0.0,
    [&](double sum, const auto& doc) { return sum + doc.size(); }) / corpus.size();
    std::cout << "Document avg len: " << avg_len << std::endl;
#ifdef __APPLE__
    print_memory_usage();
#endif
    BM25Okapi bm25(corpus);
#ifdef __APPLE__
    print_memory_usage();
#endif
    std::string input;
    while (true) {
        std::cout << "Enter a string: ";
        std::getline(std::cin, input);
        if (input == "exit") {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto scores = bm25.get_scores(split(input));
        //print_vec(scores);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "Time taken: " << diff.count() << " s\n";
    }
}

int main() {
    test_scoring();
    repl();
}
