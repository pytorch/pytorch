//TEST:EXECUTABLE:
struct SubRecord
{
    int field2;
    float field3;
}

struct Record
{
    int field;
    SubRecord sub;
}

export __extern_cpp int main()
{
    Record rec;
    Record *pRec = &rec;
    pRec.field = 1;
    pRec.sub.field2 = 2;
    pRec.sub.field3 = 3.0f;
    if (rec.field == 1 && rec.sub.field2 == 2 && pRec.sub.field3 == 3.0f)
    {
        printf("success\n");
    }
    else
    {
        printf("fail\n");
    }
    return 0;
}
