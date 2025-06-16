#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <numeric>
#include <omp.h>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }
		std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
		int curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

//pro-hnsw

	// Retrieves the maximum defined level for a given node_id.
	int getNodeMaxLevel(tableint node_id) const {
		// Ensure node_id is within the valid range of current elements.
		if (node_id >= this->cur_element_count) {
			throw std::runtime_error("getNodeMaxLevel: node_id out of range.");
		}
		// element_levels_ is assumed to store the max level for each node.
		return this->element_levels_[node_id];
	}

	// Retrieves a list of neighbor IDs for a given node_id at a specific level.
	std::vector<tableint> getNodeNeighbors(tableint node_id, int level) const {
		// Ensure node_id is within the valid range.
		if (node_id >= this->cur_element_count) {
			throw std::runtime_error("getNodeNeighbors: node_id out of range.");
		}
		// Ensure the requested level is valid for the given node_id.
		if (level < 0 || level > this->element_levels_[node_id]) {
			throw std::runtime_error("getNodeNeighbors: level out of range for this node.");
		}

		// Obtain low-level access to the node's link list at the specified level.
		linklistsizeint* ll_ptr = get_linklist_at_level(node_id, level);
		int neighbor_count = getListCount(ll_ptr);
		tableint* neighbors_data = (tableint*)(ll_ptr + 1);

		// Populate a vector with the neighbor IDs.
		std::vector<tableint> result;
		result.reserve(neighbor_count);
		for (int i = 0; i < neighbor_count; i++) {
			result.push_back(neighbors_data[i]);
		}
		return result;
	}

	// Helper function: Computes the in-degree for all nodes, considering only links at level 0.
	inline void computeAllInDegree_L0(std::vector<int>& indeg) const {
		// Ensure the indeg vector is correctly sized and initialized to zero.
		indeg.assign(this->cur_element_count, 0);

		// Iterate through all potential source nodes 'u'.
		for (tableint u = 0; u < this->cur_element_count; ++u) {
			// Skip if node 'u' is marked as deleted.
			if (isMarkedDeleted(u)) {
				continue;
			}
			
			// Get neighbors of 'u' only at level 0.
			linklistsizeint* ll = get_linklist_at_level(u, 0);
			int neighbor_list_size = getListCount(ll); // sz -> neighbor_list_size
			tableint* neighbor_data = (tableint*)(ll + 1);    // dat -> neighbor_data

			// For each neighbor 'v' of 'u' (i.e., for each edge u -> v), increment in-degree of 'v'.
			for (int i = 0; i < neighbor_list_size; i++) {
				tableint v = neighbor_data[i];
				// Increment in-degree of 'v' if 'v' is a valid and non-deleted node.
				if (v >= 0 && v < (tableint)this->cur_element_count && !isMarkedDeleted(v)) {
					++indeg[v];
				}
			}
		}
	}

	// Attempts to repair connectivity for nodes identified as part of "small" components.
	// A small component is defined as one having fewer nodes than half the total active nodes in the graph.
	// For each node in such components, it tries to establish/fill links across all its defined levels.
	size_t repairDisconnectedNodes() {
		// Retrieve all connected components based on current graph state (all levels).
		auto comps = findAllComponents();
		if (comps.empty()) {
			return 0;
		}

		// Calculate the current number of active (not deleted) nodes.
		size_t num_active_nodes = 0;
		for (tableint i = 0; i < this->cur_element_count; ++i) {
			if (!isMarkedDeleted(i)) {
				num_active_nodes++;
			}
		}

		if (num_active_nodes == 0) { // If no active nodes, no repair is possible.
			return 0;
		}

		std::unordered_set<tableint> nodes_processed_this_call; // Tracks unique nodes for which repair was attempted.

		// Threshold to identify "small" components.
		double threshold_size = num_active_nodes / 2.0;

		// Iterate through each identified component.
		for (const auto &comp : comps) {
			// Process component if it's non-empty and considered "small".
			if (comp.size() > 0 && comp.size() < threshold_size) {
				// Attempt repair for each node within this small component.
				for (auto node_id : comp) {
					if (isMarkedDeleted(node_id)) {
						continue;
					}

					bool repair_attempted_for_this_node_overall = false;
					// Get the original highest level this node was part of.
					int original_max_level_of_node = element_levels_[node_id];

					// Attempt to repair/fill links for this node_id across all its defined levels,
					// from level 0 up to its original maximum level.
					for (int level_to_repair = 0; level_to_repair <= original_max_level_of_node; ++level_to_repair) {
						// Determine the target number of neighbors (M value) for the current level.
						int target_M_for_this_level = (level_to_repair == 0) ? maxM0_ : maxM_;
						
						// Call attemptLimitedHopRepairFillM for the current node_id and level.
						// This function is assumed to attempt to add links for node_id
						// at 'level_to_repair', up to 'target_M_for_this_level' neighbors.
						// It is also assumed that attemptLimitedHopRepairFillM has been modified
						// to accept 'level_to_repair' and 'target_M_for_this_level' as arguments
						// and to operate búsqueda and adding links at that specific level.
						if (attemptLimitedHopRepairFillM(node_id, level_to_repair, target_M_for_this_level)) {
							repair_attempted_for_this_node_overall = true;
						}
					}

					if (repair_attempted_for_this_node_overall) {
						// If repair was attempted for this node at any level, mark it as processed.
						nodes_processed_this_call.insert(node_id);
					}
				}
			}
		}
		
		// Return the count of unique nodes for which a repair attempt was made.
		return nodes_processed_this_call.size();
	}

	// Finds all connected components in the graph using BFS, considering all levels.
	// Also, it logs nodes that have a total out-degree of zero across all their active levels.
	inline std::vector<std::vector<tableint>> findAllComponents() {
		std::vector<std::vector<tableint>> components;
		// visited_bfs array to keep track of visited nodes during BFS for component finding.
		std::vector<bool> visited_bfs(this->cur_element_count, false);

		// --- Part 1: Identify and log nodes with zero total out-degree ---
		// This section is primarily for diagnostics/logging and does not alter the 'components' vector.
		std::vector<tableint> nodes_with_zero_total_out_degree;
		for (tableint node_id_check = 0; node_id_check < (tableint)this->cur_element_count; ++node_id_check) {
			// Skip already deleted nodes.
			if (isMarkedDeleted(node_id_check)) {
				continue;
			}

			size_t current_node_total_out_degree = 0;
			// Consider all defined levels for the node.
			int max_level_of_node = element_levels_[node_id_check];
			for (int level = 0; level <= max_level_of_node; ++level) {
				linklistsizeint* ll_head = get_linklist_at_level(node_id_check, level);
				// getListCount() is assumed to return the current number of valid outgoing links at this level.
				current_node_total_out_degree += getListCount(ll_head);
			}

			if (current_node_total_out_degree == 0) {
				nodes_with_zero_total_out_degree.push_back(node_id_check);
			}
		}

		// Log if any nodes with zero total out-degree are found.
		// This log can be critical for diagnosing connectivity issues.
		if (!nodes_with_zero_total_out_degree.empty()) {
			std::cout << "[findAllComponents] !!! Nodes found with ZERO total out-degree across all levels: ";
			for (size_t i = 0; i < nodes_with_zero_total_out_degree.size(); ++i) {
				std::cout << nodes_with_zero_total_out_degree[i] << (i == nodes_with_zero_total_out_degree.size() - 1 ? "" : ", ");
			}
			std::cout << " (Total: " << nodes_with_zero_total_out_degree.size() << " nodes)" << std::endl;
		}
		// --- End of Part 1 ---

		// --- Part 2: Find all connected components using BFS ---
		for (tableint start_node_id = 0; start_node_id < (tableint)this->cur_element_count; ++start_node_id) {
			// Start BFS for a new component if node is not deleted and not yet visited by BFS.
			if (!isMarkedDeleted(start_node_id) && !visited_bfs[start_node_id]) {
				std::vector<tableint> current_component_nodes;
				std::queue<tableint> q;

				q.push(start_node_id);
				visited_bfs[start_node_id] = true;

				while (!q.empty()) {
					tableint current_node = q.front();
					q.pop();
					current_component_nodes.push_back(current_node);

					// Explore neighbors of current_node across all its levels.
					int max_level = element_levels_[current_node];
					for (int level = 0; level <= max_level; ++level) {
						linklistsizeint* ll_head = get_linklist_at_level(current_node, level);
						int neighbor_count = getListCount(ll_head);
						tableint* neighbors = (tableint*)(ll_head + 1);

						for (int i = 0; i < neighbor_count; ++i) {
							tableint neighbor_node = neighbors[i];
							// Add valid, non-deleted, and unvisited neighbors to the BFS queue.
							if (neighbor_node >= 0 && neighbor_node < (tableint)this->cur_element_count &&
								!isMarkedDeleted(neighbor_node) && !visited_bfs[neighbor_node]) {
								visited_bfs[neighbor_node] = true;
								q.push(neighbor_node);
							}
						}
					}
				}
				// Once BFS finished, store the found component.
				if (!current_component_nodes.empty()) {
					components.push_back(std::move(current_component_nodes));
				}
			}
		}

		// Optional: Log the total number of connected components found.
		// std::cout << "[findAllComponents] Found " << components.size() << " connected components using BFS (all levels)." << std::endl;
		return components;
	}

	// Attempts to ensure 'node_id_to_repair' acquires up to 'M_target_for_level' *incoming* links
	// at the specified 'target_level'. It explores the neighborhood of 'node_id_to_repair'
	// using BFS (traversing outgoing links at 'target_level') to find candidate neighbors ('nb_candidate').
	// An incoming link 'nb_candidate -> node_id_to_repair' is established if 'nb_candidate' has a slot.
	// The 'cur_deg' variable tracks progress towards acquiring M_target_for_level incoming links in this call.
	bool attemptLimitedHopRepairFillM(tableint node_id_to_repair, int target_level, int M_target_for_level) {
		if (isMarkedDeleted(node_id_to_repair)) {
			return false;
		}

		// 'cur_deg' is initialized with the current *outgoing* degree of node_id_to_repair at the target_level.
		// It's then used in the loop condition and incremented when an *incoming* link is formed TO node_id_to_repair.
		// This means the loop continues as long as the initial outgoing degree (or the incremented counter)
		// is less than the target M for acquiring incoming links.
		// This might be clearer if 'cur_deg' was explicitly for 'number_of_incoming_links_to_acquire_or_acquired'.
		// However, sticking to the provided code's variable usage:
		linklistsizeint* ll_node_at_target_level = get_linklist_at_level(node_id_to_repair, target_level);
		int cur_deg = getListCount(ll_node_at_target_level); // Initial out-degree of node_id_to_repair at target_level

		// If the initial out-degree itself is already >= M_target_for_level,
		// the function currently assumes no further *incoming* links need to be added via this mechanism.
		if (cur_deg >= M_target_for_level) {
			return false;
		}

		// BFS setup
		std::vector<char> visited_bfs(cur_element_count, 0);
		visited_bfs[node_id_to_repair] = 1;

		std::vector<tableint> frontier{node_id_to_repair};
		std::vector<tableint> next_frontier;

		// Tracks if any incoming link was established for node_id_to_repair during this function call.
		bool any_incoming_link_established = false;
		int hop = 0;

		// Continue BFS as long as there are nodes in the frontier and
		// the count of (initial outgoing + newly acquired incoming) links is less than M_target_for_level.
		while (!frontier.empty() && cur_deg < M_target_for_level) {
			++hop;
			// Tracks if an incoming link was added TO node_id_to_repair in the current BFS hop.
			bool incoming_link_added_this_hop = false;
			next_frontier.clear();

			// Process all nodes in the current BFS frontier.
			for (tableint u_bfs : frontier) {
				if (cur_deg >= M_target_for_level) break; // Stop if target count for incoming links is met.

				// Explore neighbors of u_bfs by traversing its outgoing links at the target_level.
				linklistsizeint* ll_u_bfs = get_linklist_at_level(u_bfs, target_level);
				int nbCnt_u_bfs = getListCount(ll_u_bfs);
				tableint* nbs_u_bfs = (tableint*)(ll_u_bfs + 1);

				for (int i = 0; i < nbCnt_u_bfs; ++i) {
					if (cur_deg >= M_target_for_level) break;

					tableint nb_candidate = nbs_u_bfs[i]; // A candidate neighbor found via u_bfs's outgoing link.
					
					// Basic validity checks for the candidate neighbor.
					if (nb_candidate < 0 || nb_candidate >= (tableint)cur_element_count || nb_candidate == node_id_to_repair) continue;
					if (visited_bfs[nb_candidate] || isMarkedDeleted(nb_candidate)) continue;
					
					visited_bfs[nb_candidate] = 1;
					next_frontier.push_back(nb_candidate);

					// Attempt to establish an *incoming* link: nb_candidate -> node_id_to_repair at target_level.
					// This modifies nb_candidate's neighbor list if it has a slot.
					if (addIncomingEdgeIfSlotOnly(nb_candidate, node_id_to_repair, target_level)) {
						incoming_link_added_this_hop = true;
						any_incoming_link_established = true;
						
						// Increment 'cur_deg' to signify that node_id_to_repair has "received" an incoming link,
						// progressing towards the M_target_for_level for this repair attempt.
						// This 'cur_deg' does not directly reflect an increase in node_id_to_repair's *outgoing* links
						// unless addIncomingEdgeIfSlotOnly also updates node_id_to_repair's outgoing links (which it doesn't per its definition).
						cur_deg++; 

						if (cur_deg >= M_target_for_level) break; // Target for incoming links met.
					}
				}
			}

			// BFS hop control: continue for at least a few hops, or stop if progress was made in this hop after min hops.
			bool force_continue_bfs = (hop < 3); // Example: ensure at least 2 full hops.
			if (!force_continue_bfs && incoming_link_added_this_hop) {
				break;
			}
			frontier.swap(next_frontier);
		}
		// Returns true if at least one *incoming* link was successfully established for node_id_to_repair at target_level.
		return any_incoming_link_established;
	}

	// Attempts to add 'node_id_to_add' to 'neigh_id's neighbor list at the specified 'level',
	// but only if 'neigh_id' has an available slot (i.e., its current neighbor count < max allowed).
	// Returns true if 'node_id_to_add' is already a neighbor or was successfully added, false otherwise.
	bool addIncomingEdgeIfSlotOnly(tableint neigh_id, tableint node_id_to_add, int level) {
		// Acquire a lock for the neighbor list of 'neigh_id' to ensure thread-safe modification.
		std::unique_lock<std::mutex> lock(this->link_list_locks_[neigh_id]);

		// Get the neighbor list pointer and current neighbor count for 'neigh_id' at 'level'.
		linklistsizeint* ll_nei = get_linklist_at_level(neigh_id, level);
		int current_neighbor_count_of_neigh = getListCount(ll_nei);
		tableint* neighbor_list_of_neigh = (tableint*)(ll_nei + 1);

		// Check if 'node_id_to_add' is already a neighbor of 'neigh_id'.
		for (int i = 0; i < current_neighbor_count_of_neigh; i++) {
			if (neighbor_list_of_neigh[i] == node_id_to_add) {
				return true; // Already connected, considered a success.
			}
		}

		// Determine the maximum number of neighbors allowed for 'neigh_id' at this 'level'.
		int M_for_this_level = (level == 0) ? this->maxM0_ : this->maxM_;
		
		// If there's an available slot in 'neigh_id's neighbor list.
		if (current_neighbor_count_of_neigh < M_for_this_level) {
			// Add 'node_id_to_add' to the list.
			neighbor_list_of_neigh[current_neighbor_count_of_neigh] = node_id_to_add;
			// Update the neighbor count for 'neigh_id'.
			setListCount(ll_nei, current_neighbor_count_of_neigh + 1);
			return true; // Successfully added.
		}

		// No available slot, do nothing.
		return false;
	}

	// Resolves edge asymmetry in the graph, typically at level 0, using OpenMP for parallelization.
	// Ensures that if an edge node_id -> neighbor_id exists, the reciprocal edge neighbor_id -> node_id
	// is also present or attempted to be added.
	size_t resolveEdgeAsymmetry() {
		// Use std::atomic for thread-safe counting of fixed edges.
		std::atomic<size_t> total_fixed_atomic(0);

		// Parallelize the loop over all nodes using OpenMP.
		#pragma omp parallel for
		for (tableint node_id = 0; node_id < this->cur_element_count; node_id++) {
			// Skip nodes that are marked as deleted.
			if (isMarkedDeleted(node_id)) {
				continue; 
			}

			// Operations are performed on level 0.
			linklistsizeint* ll_ptr = get_linklist_at_level(node_id, 0);
			int neighbor_count = getListCount(ll_ptr);

			// If the node has no neighbors at level 0, skip to the next node (performance optimization).
			if (neighbor_count == 0) {
				continue;
			}

			tableint* neighbors = (tableint*)(ll_ptr + 1);

			// Check each neighbor of node_id.
			for (int i = 0; i < neighbor_count; i++) {
				tableint neighbor_id = neighbors[i];
				if (isMarkedDeleted(neighbor_id)) {
					continue;
				}

				// Get the neighbor list of neighbor_id (also at level 0).
				linklistsizeint* neighbor_ll_ptr = get_linklist_at_level(neighbor_id, 0);
				int neighbor_neighbor_count = getListCount(neighbor_ll_ptr);
				tableint* neighbor_neighbors = (tableint*)(neighbor_ll_ptr + 1);

				// Check if a reciprocal edge (neighbor_id -> node_id) exists.
				bool reciprocal_exists = false;
				for (int j = 0; j < neighbor_neighbor_count; j++) {
					if (neighbor_neighbors[j] == node_id) {
						reciprocal_exists = true;
						break;
					}
				}

				// If the reciprocal edge does not exist, attempt to add it.
				// addIncomingEdgeIfPossible is assumed to be thread-safe for different neigh_id.
				if (!reciprocal_exists) {
					bool added = addIncomingEdgeIfPossible(neighbor_id, node_id, 0);
					if (added) {
						// Safely increment the atomic counter.
						total_fixed_atomic++;
					}
				}
			}
		}
		// Load the final count from the atomic variable.
		return total_fixed_atomic.load();
	}

	// Attempts to add 'node_id' as a neighbor to 'neigh_id' at the specified 'level'.
	// If 'neigh_id's neighbor list is full, it uses a heuristic to select the best neighbors
	// from the existing ones plus 'node_id'.
	// Returns true if 'node_id' is already a neighbor or was successfully added (either directly or via heuristic).
	bool addIncomingEdgeIfPossible(tableint neigh_id, tableint node_id_to_add, int level) { // Renamed node_id to node_id_to_add for clarity
		// Acquire a lock for 'neigh_id's neighbor list for thread-safe modification.
		std::unique_lock<std::mutex> lock(this->link_list_locks_[neigh_id]);

		linklistsizeint* ll_nei = get_linklist_at_level(neigh_id, level);
		int current_neighbor_count = getListCount(ll_nei); // szNei -> current_neighbor_count
		tableint* current_neighbors = (tableint*)(ll_nei + 1); // dataNei -> current_neighbors

		// Check if already connected.
		for (int i = 0; i < current_neighbor_count; i++) {
			if (current_neighbors[i] == node_id_to_add) {
				return true; // Already connected.
			}
		}

		// Determine the maximum number of neighbors allowed at this level.
		int M_for_level = (level == 0) ? this->maxM0_ : this->maxM_; // Mcur -> M_for_level

		// If there's an available slot, add directly.
		if (current_neighbor_count < M_for_level) {
			current_neighbors[current_neighbor_count] = node_id_to_add;
			setListCount(ll_nei, current_neighbor_count + 1);
			return true;
		} else {
			// No available slot; heuristic selection is needed.
			// Gather current neighbors and the new candidate.
			std::vector<tableint> candidates;
			candidates.reserve(current_neighbor_count + 1);
			for (int i = 0; i < current_neighbor_count; i++) {
				candidates.push_back(current_neighbors[i]);
			}
			candidates.push_back(node_id_to_add);

			// Get the data point for 'neigh_id' to be used by the heuristic.
			// getDataByInternalId is assumed to be a member function.
			const void* data_point_of_neigh = getDataByInternalId(neigh_id);
			
			// Call the heuristic function to select the best M_for_level neighbors.
			// selectNeighborsHeuristic is a key custom member function.
			std::vector<tableint> selected_neighbors =
				this->selectNeighborsHeuristic(candidates, data_point_of_neigh, M_for_level);

			// Rewrite 'neigh_id's adjacency list with the selected neighbors.
			// setListCount to 0 first, then add, is one way to manage fixed-size pre-allocated lists.
			
			// tableint* new_neighbor_data_ptr = (tableint*)(ll_nei + 1); // Redundant if current_neighbors is used.
			int new_idx = 0;
			for (const auto &nid : selected_neighbors) { // Use const auto& for range-based for
				// Ensure not writing out of allocated bounds if selected_neighbors.size() > original pre-allocation
				// This depends on how link lists are managed. Assuming M_for_level is the max capacity.
				if (new_idx < M_for_level) { // Safety check, though selected_neighbors should be <= M_for_level
					 current_neighbors[new_idx++] = nid;
				} else {
					// This case should ideally not happen if selectNeighborsHeuristic respects M_for_level.
					break; 
				}
			}
			setListCount(ll_nei, new_idx); // Set to the actual number of neighbors written.

			// Check if 'node_id_to_add' was included in the selected neighbors.
			for (const auto &nid : selected_neighbors) { // Use const auto&
				if (nid == node_id_to_add) {
					return true; // Successfully added via heuristic.
				}
			}
			return false; // Not added via heuristic.
		}
	}

	// Selects M best neighbors from a given set of candidates using a heuristic.
	// This function first calculates distances for all candidates, then applies a
	// custom heuristic (delegated to getNeighborsByHeuristic2) to select the final M neighbors.
	std::vector<tableint> selectNeighborsHeuristic(
		const std::vector<tableint> &candidates,         // Input: A list of candidate node IDs.
		const void *query_data_point,                  // Input: The data vector of the node for which neighbors are being selected.
		size_t M                                      // Input: The maximum number of neighbors to select.
	) {
		// Create a priority queue to store candidates by distance.
		// CompareByFirst is a custom comparator determining the queue's ordering
		// (e.g., a min-priority queue storing pairs of <distance, node_id>).
		std::priority_queue<std::pair<dist_t, tableint>,
							  std::vector<std::pair<dist_t, tableint>>,
							  CompareByFirst>
							  candidate_priority_queue; // Stores <distance, candidate_id>

		// Calculate distances for all candidates and populate the priority queue.
		for (auto candidate_id : candidates) {
			// getDataByInternalId retrieves the data vector for the candidate_id.
			// fstdistfunc_ is the distance function used (e.g., L2, cosine).
			dist_t distance = fstdistfunc_(getDataByInternalId(candidate_id), query_data_point, dist_func_param_);
			candidate_priority_queue.emplace(distance, candidate_id);
		}

		// Apply the main heuristic via getNeighborsByHeuristic2.
		// This member function is expected to modify candidate_priority_queue in-place,
		// pruning it to contain the M best neighbors according to the heuristic.
		this->getNeighborsByHeuristic2(candidate_priority_queue, M);

		// Extract the selected neighbor IDs from the (now pruned) priority queue.
		std::vector<tableint> selected_neighbors;
		selected_neighbors.reserve(M); // Reserve space for up to M neighbors.
		while (!candidate_priority_queue.empty()) {
			selected_neighbors.push_back(candidate_priority_queue.top().second); // Get the node ID
			candidate_priority_queue.pop();
		}
		// The order of elements in selected_neighbors depends on the nature of CompareByFirst
		// and how getNeighborsByHeuristic2 structures the priority queue before extraction.
		return selected_neighbors;
	}

	// Removes edges pointing to deleted nodes from active nodes' neighbor lists.
	// Conditionally preserves dead edges if an active node's alive neighbor count
	// at a specific level falls below MIN_ALIVE_NEIGHBORS_TO_KEEP.
	size_t removeObsoleteEdges(const std::vector<tableint> &delete_nodes) {
		std::unordered_set<tableint> delSet(delete_nodes.begin(), delete_nodes.end());
		size_t total_dead_edges_actually_removed = 0;

		// Minimum number of alive neighbors to retain per node per level.
		// If actual alive neighbors are fewer, dead edges on that list are not removed in this pass.
		const int MIN_ALIVE_NEIGHBORS_TO_KEEP = 1;

		for (tableint u = 0; u < this->cur_element_count; u++) {
			if (this->isMarkedDeleted(u)) {
				continue;
			}

			int maxLevel = this->element_levels_[u];
			for (int level = 0; level <= maxLevel; level++) {
				linklistsizeint* ll_ptr = get_linklist_at_level(u, level);
				int original_nbCount = getListCount(ll_ptr);

				if (original_nbCount == 0) {
					continue;
				}

				tableint* neighbors_data_ptr = (tableint*)(ll_ptr + 1);

				// Calculate current number of alive neighbors for (u, level).
				int current_alive_neighbor_count = 0;
				for (int i = 0; i < original_nbCount; i++) {
					if (delSet.find(neighbors_data_ptr[i]) == delSet.end()) {
						current_alive_neighbor_count++;
					}
				}

				// If alive neighbor count is below threshold, skip dead edge removal for this list
				// to maintain some (even if dead) outgoing links.
				if (current_alive_neighbor_count < MIN_ALIVE_NEIGHBORS_TO_KEEP) {
					continue;
				}

				// Proceed to remove dead edges if there are enough alive neighbors.
				int write_pos = 0;
				for (int i = 0; i < original_nbCount; i++) {
					tableint current_neighbor = neighbors_data_ptr[i];

					if (delSet.find(current_neighbor) == delSet.end()) { // Alive neighbor
						if (i != write_pos) {
							neighbors_data_ptr[write_pos] = current_neighbor;
						}
						write_pos++;
					} else { // Dead neighbor (to be removed)
						total_dead_edges_actually_removed++;
					}
				}
				
				if (write_pos < original_nbCount) { // If any edges were removed
					setListCount(ll_ptr, write_pos);
				}
			}
		}
		return total_dead_edges_actually_removed;
	}

//end of pro-hnsw

};
}  // namespace hnswlib
