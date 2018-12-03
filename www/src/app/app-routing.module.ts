import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ClassifierComponent } from './classifier/classifier.component';
import { HomeComponent } from './home/home.component';
import { DataComponent } from './data/data.component';
import { TrainComponent } from './train/train.component';
import { TestComponent } from './test/test.component';

const routes: Routes = [
  { path : '', component: HomeComponent },
  { path : 'data', component: DataComponent },
  { path : 'train', component: TrainComponent },
  { path : 'test', component: TestComponent },
  { path : 'classifier', component: ClassifierComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { enableTracing : true })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
