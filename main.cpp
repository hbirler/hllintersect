#include <iostream>
#include <array>
#include <numeric>
#include <bit>
#include <vector>
#include <span>
#include <unordered_map>
#include "wyhash.h"

using namespace std;

/// Based on "New cardinality estimation algorithms for HyperLogLog sketches"
struct HLL {
    static constexpr unsigned p = 6;
    static constexpr unsigned q = 64 - p;
    static constexpr unsigned n = (1 << p);
    std::array<uint8_t, n> K{};


    void add(uint64_t hash) {
        auto reg = hash >> q;
        uint8_t v = countl_zero((hash << p) | (1ull << (p - 1))) + 1;
        K[reg] = max(K[reg], v);
    }

    double estimate() const;
    void merge(const HLL& b);
    static HLL merge(const HLL& a, const HLL& b);
    static double intersect(span<const HLL> hlls);
};

HLL HLL::merge(const HLL&a, const HLL&b) {
    HLL result{a};
    result.merge(b);
    return result;
}

void HLL::merge(const HLL& b) {
    for (size_t i = 0; i < n; i++)
        K[i] = max(b.K[i], K[i]);
}

static constexpr double sigma(double x) {
    if (x == 1) return numeric_limits<double>::infinity();
    double y = 1;
    double z = x;
    double zp;
    do {
        x = x * x;
        zp = z;
        z = z + x * y;
        y = 2 * y;
    } while (zp != z);
    return z;
}

static double tau(double x) {
    if (x == 0 || x == 1) return 0;
    double y = 1;
    double z = 1 - x;
    double zp;
    do {
        x = sqrt(x);
        y = 0.5 * y;
        zp = z;
        z = z - (1 - x) * (1 - x) * y;
    } while (z != zp);
    return z / 3.0;
}

double HLL::estimate() const {
    array<unsigned, q + 2> C{};
    for (size_t i = 0; i < n; i++)
        C[K[i]]++;
    double m = accumulate(C.begin(), C.end(), 0u);
    double z = m * tau(1 - C[q + 1] / m);
    for (size_t k = q; k > 0; k--)
        z = 0.5 * (z + C[k]);
    z = z + m * sigma(C[0] / m);

    // 1/(2log2)
    constexpr double alphaInf = 0.7213475204444817036799623405009460687133229770764929670677247034;
    return alphaInf * m * m / z;
}

double HLL::intersect(span<const HLL> hlls) {
    if (hlls.empty())
        return 0.0;
    if (hlls.size() == 1)
        return hlls[0].estimate();
    vector<HLL> unions(1ull << hlls.size());
    for (size_t i = 0; i < hlls.size(); i++)
        unions[1ull << i] = hlls[i];
    // Compute all unions
    for (size_t i = 1; i < unions.size(); i++) {
        auto lsb = i & (-i);
        // Not sure this is the best statistical way
        unions[i] = HLL::merge(unions[lsb], unions[i - lsb]);
    }

    // Inclusion exclusion
    double result = 0.0;
    for (size_t i = 1; i < unions.size(); i++) {
        double mult = (popcount(i) % 2 == 0) ? -1 : 1;
        result += mult * unions[i].estimate();
    }
    return result;
};

static constexpr int64_t Low = numeric_limits<int32_t>::min();
static constexpr int64_t High = numeric_limits<int32_t>::max();

// Pretty inefficient segment tree:
// * Recursion could easily be avoided in add (it isn't)
// * Additionally, nodes could contain more than one element (they don't)
struct ST {
    struct Node {
        HLL value;
        unique_ptr<Node> left, right;
    };
    unique_ptr<Node> root;

    HLL findInternal(int64_t low, int64_t high, const Node* node, int64_t nodeLow, int64_t nodeHigh) const {
        if (!node || (high < nodeLow) || (low > nodeHigh) || (nodeLow > nodeHigh))
            return {};
        if ((high >= nodeHigh) && (low <= nodeLow))
            return node->value;
        auto mid = midpoint(nodeLow, nodeHigh);
        auto left = findInternal(low, high, node->left.get(), nodeLow, mid);
        auto righ = findInternal(low, high, node->right.get(), mid + 1, nodeHigh);
        left.merge(righ);
        return left;
    }

    HLL find(int64_t low, int64_t high = High) const {
        return findInternal(low, high, root.get(), Low, High);
    }

    void addInternal(int64_t key, uint64_t hash, unique_ptr<Node>& node, int64_t nodeLow, int64_t nodeHigh) {
        if ((key < nodeLow) || (key > nodeHigh) || (nodeLow > nodeHigh))
            return;
        if (!node)
            node = make_unique<Node>();
        node->value.add(hash);
        if ((key == nodeLow) && (key == nodeHigh))
            return;
        auto mid = midpoint(nodeLow, nodeHigh);
        if (key <= mid) {
            return addInternal(key, hash, node->left, nodeLow, mid);
        } else {
            return addInternal(key, hash, node->right, mid + 1, nodeHigh);
        }
    }

    void add(int64_t key, uint64_t value) {
        return addInternal(key, wyhash64(0, value), root, Low, High);
    }

};

// We assume attribute 0 is the cost
struct DB {
    static constexpr unsigned CostAttr = 0;
    unordered_map<unsigned, ST> indices;

    // Add an attribute with value to an item
    void addAttribute(unsigned itemId, unsigned attrId, unsigned value) {
        indices[attrId].add(value, itemId);
    }

    HLL getHLL(unsigned attrId, int64_t low, int64_t high) {
        auto it = indices.find(attrId);
        if (it == indices.end())
            return {};
        return it->second.find(low, high);
    }

    struct Restriction {
        unsigned attrId = 0;
        int64_t low = Low;
        int64_t high = High;
    };

    // Estimate the count given restrictions
    double estimateCount(span<const Restriction> restrictions, int64_t costLow, int64_t costHigh) {
        vector<HLL> hlls;
        hlls.reserve(restrictions.size() + 1);
        for (const auto& r : restrictions)
            hlls.push_back(getHLL(r.attrId, r.low, r.high));
        hlls.push_back(getHLL(CostAttr, costLow, costHigh));

        return HLL::intersect(hlls);
    }

    // Estimate the cost of the kth item with the given restrictions
    int64_t estimateCost(span<const Restriction> restrictions, unsigned k = 30) {
        // Binary search on the cost
        int64_t low = Low;
        int64_t high = High;

        while (low < high) {
            auto mid = midpoint(low, high);
            // Cost should be <= mid
            double count = estimateCount(restrictions, Low, mid);
            if (count > k)
                high = mid - 1;
            else
                low = mid + 1;
        }
        return low;
    }
};

int main() {
    cout << "Hello, World!" << endl;

    DB db;
    for (size_t i = 0; i < 100; i++) {
        // To item i, add attribute 0 with value i + 40
        // Attribute 0 is cost
        db.addAttribute(i, 0, i + 40);
        // To item i, add attribute 1 with value i + 40
        db.addAttribute(i, 1, i + 100);
    }

    double res = db.estimateCost(vector<DB::Restriction>{{1, 150, 300}});
    cout << res << endl;
    return 0;
}
