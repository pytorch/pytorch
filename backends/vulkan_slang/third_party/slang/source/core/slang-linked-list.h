#ifndef SLANG_CORE_LINKED_LIST_H
#define SLANG_CORE_LINKED_LIST_H

#include "slang-allocator.h"
#include "slang-list.h"
#include "slang.h"

#include <type_traits>

namespace Slang
{
template<typename T>
class LinkedList;

template<typename T>
class LinkedNode
{
    template<typename T1>
    friend class LinkedList;

private:
    LinkedNode<T>* prev = nullptr;
    LinkedNode<T>* next = nullptr;
    LinkedList<T>* list;

public:
    T value;
    LinkedNode(LinkedList<T>* lnk)
        : list(lnk){};
    LinkedNode<T>* getPrevious() { return prev; };
    LinkedNode<T>* getNext() { return next; };
    const LinkedNode<T>* getNext() const { return next; };
    LinkedNode<T>* insertAfter(const T& nData)
    {
        LinkedNode<T>* n = new LinkedNode<T>(list);
        n->value = nData;
        n->prev = this;
        n->next = this->next;
        LinkedNode<T>* npp = n->next;
        if (npp)
        {
            npp->prev = n;
        }
        next = n;
        if (!n->next)
            list->tail = n;
        list->count++;
        return n;
    };
    LinkedNode<T>* insertBefore(const T& nData)
    {
        LinkedNode<T>* n = new LinkedNode<T>(list);
        n->value = nData;
        n->prev = prev;
        n->next = this;
        prev = n;
        LinkedNode<T>* npp = n->prev;
        if (npp)
            npp->next = n;
        if (!n->prev)
            list->head = n;
        list->count++;
        return n;
    };
    void removeAndDelete()
    {
        if (prev)
            prev->next = next;
        if (next)
            next->prev = prev;
        list->count--;
        if (list->head == this)
        {
            list->head = next;
        }
        if (list->tail == this)
        {
            list->tail = prev;
        }
        delete this;
    }
};

template<typename T>
class LinkedList
{
    template<typename T1>
    friend class LinkedNode;

private:
    LinkedNode<T>*head, *tail;
    int count;

public:
    template<bool Const>
    class GenIterator
    {
    public:
        using Node = std::conditional_t<Const, const LinkedNode<T>, LinkedNode<T>>;
        Node *current, *next;
        void setCurrent(Node* cur)
        {
            current = cur;
            if (current)
                next = current->getNext();
            else
                next = nullptr;
        }
        GenIterator() { current = next = nullptr; }
        GenIterator(Node* cur) { setCurrent(cur); }
        std::conditional_t<Const, const T&, T&> operator*() const { return current->value; }
        GenIterator& operator++()
        {
            setCurrent(next);
            return *this;
        }
        GenIterator operator++(int)
        {
            GenIterator rs = *this;
            setCurrent(next);
            return rs;
        }
        bool operator!=(const GenIterator& iter) const { return current != iter.current; }
        bool operator==(const GenIterator& iter) const { return current == iter.current; }
    };

    using Iterator = GenIterator<false>;
    Iterator begin() { return Iterator(head); }
    Iterator end() { return Iterator(0); }

    using ConstIterator = GenIterator<true>;
    ConstIterator begin() const { return ConstIterator(head); }
    ConstIterator end() const { return ConstIterator(0); }

public:
    LinkedList()
        : head(0), tail(0), count(0)
    {
    }
    ~LinkedList() { clear(); }
    LinkedList(const LinkedList<T>& link)
        : head(0), tail(0), count(0)
    {
        this->operator=(link);
    }
    LinkedList(LinkedList<T>&& link)
        : head(0), tail(0), count(0)
    {
        this->operator=(_Move(link));
    }
    LinkedList<T>& operator=(LinkedList<T>&& link)
    {
        if (head != 0)
            clear();
        head = link.head;
        tail = link.tail;
        count = link.count;
        link.head = 0;
        link.tail = 0;
        link.count = 0;
        for (auto node = head; node; node = node->getNext())
            node->list = this;
        return *this;
    }
    LinkedList<T>& operator=(const LinkedList<T>& link)
    {
        if (head != nullptr)
            clear();
        auto p = link.head;
        while (p)
        {
            addLast(p->value);
            p = p->getNext();
        }
        return *this;
    }
    template<typename IteratorFunc>
    void forEach(const IteratorFunc& f)
    {
        auto p = head;
        while (p)
        {
            f(p->value);
            p = p->getNext();
        }
    }
    LinkedNode<T>* getNode(int x)
    {
        LinkedNode<T>* pCur = head;
        for (int i = 0; i < x; i++)
        {
            if (pCur)
                pCur = pCur->next;
            else
                SLANG_UNEXPECTED("Index out of range");
        }
        return pCur;
    };
    LinkedNode<T>* find(const T& fData)
    {
        for (LinkedNode<T>* pCur = head; pCur; pCur = pCur->next)
        {
            if (pCur->value == fData)
                return pCur;
        }
        return nullptr;
    };
    LinkedNode<T>* getFirstNode() const { return head; };
    T& getFirst() const
    {
        if (!head)
            SLANG_UNEXPECTED("LinkedList: index out of range.");
        return head->value;
    }
    T& getLast() const
    {
        if (!tail)
            SLANG_UNEXPECTED("LinkedList: index out of range.");
        return tail->value;
    }
    LinkedNode<T>* getLastNode() const { return tail; };
    LinkedNode<T>* addLast(const T& nData)
    {
        LinkedNode<T>* n = new LinkedNode<T>(this);
        n->value = nData;
        n->prev = tail;
        if (tail)
            tail->next = n;
        n->next = 0;
        tail = n;
        if (!head)
            head = n;
        count++;
        return n;
    };
    // Insert a blank node
    LinkedNode<T>* addLast()
    {
        LinkedNode<T>* n = new LinkedNode<T>(this);
        n->prev = tail;
        if (tail)
            tail->next = n;
        n->next = 0;
        tail = n;
        if (!head)
            head = n;
        count++;
        return n;
    };
    LinkedNode<T>* addFirst(const T& nData)
    {
        LinkedNode<T>* n = new LinkedNode<T>(this);
        n->value = nData;
        addFirst(n);
        count++;
        return n;
    };
    void addFirst(LinkedNode<T>* n)
    {
        n->prev = 0;
        n->next = head;
        if (head)
            head->prev = n;
        head = n;
        if (!tail)
            tail = n;
    }
    void removeFromList(LinkedNode<T>* n)
    {
        LinkedNode<T>*n1, *n2 = 0;
        n1 = n->prev;
        n2 = n->next;
        if (n1)
            n1->next = n2;
        else
            head = n2;
        if (n2)
            n2->prev = n1;
        else
            tail = n1;
        n->prev = nullptr;
        n->next = nullptr;
    }
    void removeAndDelete(LinkedNode<T>* n, int Count = 1)
    {
        LinkedNode<T>*cur, *next;
        cur = n;
        int numDeleted = 0;
        for (int i = 0; i < Count; i++)
        {
            next = cur->next;
            removeFromList(cur);
            delete cur;
            cur = next;
            numDeleted++;
            if (cur == 0)
                break;
        }
        count -= numDeleted;
    }
    void clear()
    {
        for (LinkedNode<T>* n = head; n;)
        {
            LinkedNode<T>* tmp = n->next;
            delete n;
            n = tmp;
        }
        head = 0;
        tail = 0;
        count = 0;
    }
    List<T> toList() const
    {
        List<T> rs;
        rs.Reserve(count);
        for (auto& item : *this)
        {
            rs.add(item);
        }
        return rs;
    }
    int getCount() const { return count; }
};
} // namespace Slang
#endif
