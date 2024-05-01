class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다
        dy = dout * self.x

        return dx, dy


class AddLayer:

    # 덧셈을 값을 저장할 필요 X
    # 그냥 배분
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


"""
# apple * apple_num * tax
apple = 100
apple_num = 2
tax = 1.1

# layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)  # 220

# backward
dprice = 1  # apple/apple_num/tax가 각각 1만큼 오를 때 가격에 어떤 영향을 미치는지
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)  # 2.2, 110, 200
"""

#####################################################################################

# ((apple * apple_num) + (orange * orange_num)) * tax
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layers
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(apple_orange_price, tax)

print(total_price)  # 715

# backward
dtotal_price = 1  # 각각이 1만큼 오를 때 가격에 어떤 영향을 미치는지
dapple_orange_price, dtax = mul_tax_layer.backward(dtotal_price)
dapple_price, dorange_price = add_apple_orange_layer.backward(dapple_orange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print(dapple, dapple_num, dorange, dorange_num, dtax)  # 2.2, 110, 3.3, 165, 650
