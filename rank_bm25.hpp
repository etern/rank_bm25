#ifndef RANK_BM25_HPP
#define RANK_BM25_HPP
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <numeric>

class BM25 {
protected:
    int corpus_size;
    double avgdl;
    std::vector<std::unordered_map<std::string, int>> doc_freqs;
    std::unordered_map<std::string, double> idf_;
    std::vector<int> doc_len;

    void initialize(const std::vector<std::vector<std::string>>& corpus) {
        std::unordered_map<std::string, int> nd;  // word -> number of documents with word
        int num_doc = 0;
        for (const auto& document : corpus) {
            doc_len.push_back(document.size());
            num_doc += document.size();

            std::unordered_map<std::string, int> frequencies;
            for (const auto& word : document) {
                if (frequencies.find(word) == frequencies.end()) {
                    frequencies[word] = 0;
                }
                frequencies[word] += 1;
            }
            doc_freqs.push_back(frequencies);

            for (const auto& word_freq : frequencies) {
                if (nd.find(word_freq.first) == nd.end()) {
                    nd[word_freq.first] = 0;
                }
                nd[word_freq.first] += 1;
            }

            corpus_size += 1;
        }

        avgdl = static_cast<double>(num_doc) / corpus_size;
        calc_idf(nd);
    }

    virtual void calc_idf(const std::unordered_map<std::string, int>& nd) = 0;
    virtual std::vector<double> get_scores(const std::vector<std::string>& query) = 0;
public:
    BM25():corpus_size(0), avgdl(0) {
    }
    std::vector<int> get_top_n(const std::vector<std::string>& query, int n=5) {
        auto scores = get_scores(query);
        std::vector<int> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        n = std::min({n, (int)indices.size()});
        std::partial_sort(indices.begin(), indices.begin() + n, indices.end(),
            [&scores](const int i1, const int i2) { return scores[i1] > scores[i2]; });

        std::vector<int> top_n(indices.begin(), indices.begin() + n);
        return top_n;
    }
};

class BM25Okapi : public BM25 {
private:
    double k1;
    double b;
    double epsilon;

    void calc_idf(const std::unordered_map<std::string, int>& nd) override {
        double idf_sum = 0;
        std::vector<std::string> negative_idfs;
        for (const auto& word_freq : nd) {
            double idf = std::log(corpus_size - word_freq.second + 0.5) - std::log(word_freq.second + 0.5);
            idf_[word_freq.first] = idf;
            idf_sum += idf;
            if (idf < 0) {
                negative_idfs.push_back(word_freq.first);
            }
        }
        double average_idf = idf_sum / idf_.size();

        double eps = epsilon * average_idf;
        for (const auto& word : negative_idfs) {
            idf_[word] = eps;
        }
    }

public:
    BM25Okapi(const std::vector<std::vector<std::string>>& corpus, double k1 = 1.5, double b = 0.75, double epsilon = 0.25)
        : k1(k1), b(b), epsilon(epsilon) {
        initialize(corpus);
    }

    std::vector<double> get_scores(const std::vector<std::string>& query) {
        std::vector<double> scores(corpus_size, 0.0);
        for (const auto& q : query) {
            auto q_idf = idf_.find(q);
            if (q_idf != idf_.end()) {
                for (size_t i = 0; i < corpus_size; ++i) {
                    auto q_freq = doc_freqs[i].find(q);
                    if (q_freq != doc_freqs[i].end()) {
                        double term_score = (q_idf->second * (q_freq->second * (k1 + 1))) /
                                            (q_freq->second + k1 * (1 - b + b * doc_len[i] / avgdl));
                        scores[i] += term_score;
                    }
                }
            }
        }
        return scores;
    }
};

class BM25L : public BM25 {
private:
    double k1;
    double b;
    double delta;
    double average_idf;
    void calc_idf(const std::unordered_map<std::string, int>& nd) override {
        for (const auto& word_freq : nd) {
            double idf = std::log(corpus_size + 1) - std::log(word_freq.second + 0.5);
            idf_[word_freq.first] = idf;
        }
    }
public:
    BM25L(const std::vector<std::vector<std::string>>& corpus, double k1 = 1.5, double b = 0.75, double delta = 0.5)
    : k1(k1), b(b), delta(delta), average_idf(0) {
        initialize(corpus);
    }

    std::vector<double> get_scores(const std::vector<std::string>& query) {
        std::vector<double> score(corpus_size, 0.0);
        for (const auto& q : query) {
            for (int i = 0; i < corpus_size; ++i) {
                double q_freq = doc_freqs[i][q];
                double ctd = q_freq / (1 - b + b * doc_len[i] / avgdl);
                score[i] += (idf_[q] * (k1 + 1) * (ctd + delta)) / (k1 + ctd + delta);
            }
        }
        return score;
    }
};

class BM25Plus : public BM25 {
private:
    double k1;
    double b;
    double delta;
    double average_idf;
    void calc_idf(const std::unordered_map<std::string, int>& nd) override {
        for (const auto& word_freq : nd) {
            double idf = std::log((corpus_size + 1) / static_cast<double>(word_freq.second));
            idf_[word_freq.first] = idf;
        }
    }
public:
    BM25Plus(const std::vector<std::vector<std::string>>& corpus, double k1 = 1.5, double b = 0.75, double delta = 1)
    : k1(k1), b(b), delta(delta), average_idf(0) {
        initialize(corpus);
    }
    std::vector<double> get_scores(const std::vector<std::string>& query) {
        std::vector<double> score(corpus_size, 0.0);
        for (const auto& q : query) {
            std::vector<int> q_freq(corpus_size);
            for (int i = 0; i < corpus_size; ++i) {
                q_freq[i] = doc_freqs[i].count(q) ? doc_freqs[i][q] : 0;
            }
            for (int i = 0; i < corpus_size; ++i) {
                double idf_q = idf_.count(q) ? idf_[q] : 0.0;
                score[i] += idf_q * (delta + (q_freq[i] * (k1 + 1)) /
                    (k1 * (1 - b + b * doc_len[i] / avgdl) + q_freq[i]));
            }
        }
        return score;
    }
};
#endif // RANK_BM25_HPP
