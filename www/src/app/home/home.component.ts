import { Component, OnInit } from '@angular/core';
import { ModelService } from '../model.service';
import { Model } from '../model';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.less']
})
export class HomeComponent implements OnInit {
  models: Model[];

  constructor(private modelService : ModelService) { }

  ngOnInit() {
    this.getModels();
  }

  getModels(): void {
    this.modelService.getModels()
      .subscribe(models => this.models = models);
  }

}
