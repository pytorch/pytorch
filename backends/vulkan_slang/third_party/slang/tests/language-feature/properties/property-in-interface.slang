// property-in-interface.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj -vk

// Test that interfaces can include property declarations.

interface ICell
{
    property value : int { get; set; }
}

struct MyCell : ICell
{
    var _data : int;

    property value : int { get { return _data; } set(newValue) { _data = newValue; } }
}

struct YourCell : ICell
{
    int value;

    int getValue() { return value; }

    [mutating] void setValue(int v) { value = v; }
}

int helper<C : ICell>(C cell)
{
    cell.value = cell.value + 1;
    return cell.value;
}

int test(int value)
{
    MyCell myCell = { value+1 };
    YourCell yourCell = { value };

    // Note: fetching `value` directly from `YourCell`
    // to confirm that member lookup is prioritizing
    // the concrete `YourCell::value` member of the inherited
    // abstract member `ICell::value`.
    //
    int f = (yourCell.value + yourCell.getValue()) / 2;
    return helper(myCell)*16 + helper(yourCell) + f;
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int inVal = outputBuffer[tid];
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
